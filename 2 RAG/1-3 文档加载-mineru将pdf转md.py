import os
import shutil
import time
import zipfile
import requests
from dotenv import load_dotenv

load_dotenv(override=True)

BASE = "https://mineru.net/api/v4"
TOKEN = os.getenv("mineru_api_token")


def upload_pdfs(
        pdf_paths,
        token,
        model_version="vlm",
        page_ranges=None,
        is_ocr=False,
        enable_formula=True,
        enable_table=True,
        language="ch",
        extra_formats=None,
        data_ids=None,
):
    """
    批量上传 PDF 并自动提交解析任务（单批 ≤50 个）。
    会打印每个文件对应的上传链接。
    """
    if len(pdf_paths) > 50:
        raise ValueError("单次申请上传链接不能超过 50 个文件")
    for p in pdf_paths:
        if not p.lower().endswith(".pdf"):
            raise ValueError(f"仅支持 PDF: {p}")

    header = {"Content-Type": "application/json", "Authorization": f"Bearer {token}"}

    files_meta = []
    for i, fp in enumerate(pdf_paths):
        name = os.path.basename(fp)
        meta = {"name": name}
        if is_ocr:
            meta["is_ocr"] = True
        if page_ranges and name in page_ranges:
            meta["page_ranges"] = page_ranges[name]
        if data_ids and i < len(data_ids):
            meta["data_id"] = data_ids[i]
        files_meta.append(meta)

    data = {
        "files": files_meta,
        "model_version": model_version,
        "enable_formula": enable_formula,
        "enable_table": enable_table,
        "language": language,
    }
    if extra_formats:
        data["extra_formats"] = extra_formats

    resp = requests.post(f"{BASE}/file-urls/batch", headers=header, json=data, timeout=60)
    result = resp.json()
    if resp.status_code != 200 or result.get("code") != 0:
        print(f"申请上传链接失败: {result.get('msg')}")
        return None

    batch_id = result["data"]["batch_id"]
    urls = result["data"]["file_urls"]
    print(f"batch_id: {batch_id}，共 {len(urls)} 个上传链接")

    uploaded, failed = [], []
    for i, url in enumerate(urls):
        name = files_meta[i]["name"]
        print(f"[{i + 1}/{len(urls)}] {name} 上传链接: {url}")
        try:
            # 上传时不要设置 Content-Type
            with open(pdf_paths[i], "rb") as f:
                r = requests.put(url, data=f, timeout=300)
            if r.status_code in (200, 201):
                uploaded.append(name)
                print(f"    -> 上传成功")
            else:
                failed.append((name, f"HTTP {r.status_code}"))
                print(f"    -> 上传失败: HTTP {r.status_code}")
        except Exception as e:
            failed.append((name, str(e)))
            print(f"    -> 上传异常: {e}")

    return {"batch_id": batch_id, "uploaded": uploaded, "failed": failed}


def get_pdf_results(batch_id, token, timeout=1800, interval=5, wait=True):
    """
    批量获取解析结果，打印每个文件的下载链接（full_zip_url）。
    """
    header = {"Content-Type": "application/json", "Authorization": f"Bearer {token}"}
    url = f"{BASE}/extract-results/batch/{batch_id}"
    start = time.time()

    while True:
        resp = requests.get(url, headers=header, timeout=60)
        result = resp.json()
        if resp.status_code != 200 or result.get("code") != 0:
            print(f"查询失败: {result.get('msg')}")
            return None

        items = result["data"].get("extract_result", [])
        results, all_done = [], True
        for it in items:
            state = it.get("state")
            if state not in ("done", "failed"):
                all_done = False
            progress = it.get("extract_progress")
            if progress:
                progress = f"{progress.get('extracted_pages')}/{progress.get('total_pages')}"
            results.append({
                "file_name": it.get("file_name"),
                "state": state,
                "full_zip_url": it.get("full_zip_url"),
                "err_msg": it.get("err_msg", ""),
                "data_id": it.get("data_id"),
                "progress": progress,
            })

        if not wait or all_done:
            print("\n===== 解析结果 =====")
            for r in results:
                line = f"{r['file_name']}: state={r['state']}"
                if r["state"] == "done" and r["full_zip_url"]:
                    line += f"\n    下载链接: {r['full_zip_url']}"
                elif r["state"] == "failed":
                    line += f"  err_msg={r['err_msg']}"
                elif r["progress"]:
                    line += f"  progress={r['progress']}"
                print(line)
            return {"batch_id": batch_id, "all_done": all_done, "results": results}

        if time.time() - start > timeout:
            print(f"轮询超时（{timeout}s），请稍后手动查询 batch_id: {batch_id}")
            return {"batch_id": batch_id, "all_done": False, "results": results}

        elapsed = int(time.time() - start)
        done_cnt = sum(1 for r in results if r["state"] in ("done", "failed"))
        print(f"[{elapsed}s] 已完成 {done_cnt}/{len(results)}，等待中...")
        time.sleep(interval)


def _safe_extract(zf, target_dir):
    """解压并防止 zip slip（.. 或绝对路径穿透）。"""
    for member in zf.infolist():
        member_path = os.path.abspath(os.path.join(target_dir, member.filename))
        if not member_path.startswith(os.path.abspath(target_dir) + os.sep):
            print(f"跳过可疑路径: {member.filename}")
            continue
        zf.extract(member, target_dir)


def download_all(results, out_dir="./mineru_out"):
    """
    下载每个 full_zip_url，并解压保留 zip 中的全部产物。

    目录结构:
        out_dir/<pdf文件名>/
            full.md
            *_content_list.json
            *_model.json
            layout.json
            images/...
            （若 extra_formats 含 docx/html/latex 也会一并保留）

    返回:
        [(file_name, save_dir_or_None, err)]
    """
    os.makedirs(out_dir, exist_ok=True)
    summary = []
    for r in results:
        name, state, zip_url = r["file_name"], r["state"], r["full_zip_url"]
        stem = os.path.splitext(name)[0]
        if state != "done" or not zip_url:
            summary.append((name, None, f"state={state}, {r.get('err_msg', '')}"))
            print(f"{name}: 跳过（state={state}）")
            continue

        save_dir = os.path.join(out_dir, stem)
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        os.makedirs(save_dir, exist_ok=True)

        print(f"{name} 下载中: {zip_url}")
        try:
            zresp = requests.get(zip_url, timeout=300)
            zresp.raise_for_status()
            zip_path = os.path.join(save_dir, f"{stem}.zip")
            with open(zip_path, "wb") as f:
                f.write(zresp.content)

            with zipfile.ZipFile(zip_path) as zf:
                _safe_extract(zf, save_dir)

            # 校验是否有 full.md（可选，仅供参考）
            has_md = any(
                os.path.isfile(os.path.join(dp, fn))
                for dp, _, fns in os.walk(save_dir) for fn in fns if fn == "full.md"
            )
            summary.append((name, save_dir, "" if has_md else "未找到 full.md，但其他产物已保留"))
            print(f"    -> 已保存到: {save_dir}")

        except Exception as e:
            summary.append((name, None, str(e)))
            print(f"    -> 下载/解压失败: {e}")
    return summary


def pdf_to_md(pdf_paths, out_dir):
    # 1 发送请求，获取上传链接和 batchid
    # 2 再发送请求，根据上传链接上传文件
    # 3 轮询发送请求，根据 batchid 获取下载地址
    # 4 下载
    up = upload_pdfs(
        pdf_paths,
        TOKEN
    )
    if up:
        res = get_pdf_results(up["batch_id"], TOKEN)
        download_all(res["results"], out_dir=out_dir)


if __name__ == "__main__":
    pdf_to_md(pdf_paths=["docs/中国科学院国家天文台2023年部门预算.pdf"], out_dir="docs")

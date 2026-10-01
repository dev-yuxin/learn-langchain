import os
import time

import requests
from dotenv import load_dotenv

load_dotenv(override=True)

BASE_URL = "https://mineru.net/api/v4"


def _get_token() -> str:
    """从环境变量读取 MinerU API Token。"""
    token = os.getenv("MINERU_API_KEY")
    if not token:
        raise OSError("未找到环境变量 MINERU_API_KEY，请先设置再运行。")
    return token


def _headers() -> dict:
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {_get_token()}",
    }


# ── 1. 申请上传链接 ──────────────────────────────────
def get_upload_urls(
    files: list[dict],
    model_version: str = "vlm",
    enable_formula: bool = True,
    enable_table: bool = True,
    language: str = "ch",
) -> dict:
    """
    批量申请文件上传链接。

    :param files: [{"name": "xxx.pdf", "data_id": "id1"}, ...]
    :param model_version: pipeline / vlm / MinerU-HTML
    :return: {"batch_id": str, "file_urls": [str, ...]}
    """
    url = f"{BASE_URL}/file-urls/batch"
    payload = {
        "files": files,
        "model_version": model_version,
        "enable_formula": enable_formula,
        "enable_table": enable_table,
        "language": language,
    }

    resp = requests.post(url, headers=_headers(), json=payload, timeout=30)
    resp.raise_for_status()
    result = resp.json()

    if result.get("code") != 0:
        raise RuntimeError(f"申请上传链接失败: {result.get('msg')}")

    return {
        "batch_id": result["data"]["batch_id"],
        "file_urls": result["data"]["file_urls"],
    }


# ── 2. 上传文件到预签名 URL ──────────────────────────
def upload_file(upload_url: str, local_path: str) -> bool:
    """将单个本地文件 PUT 到预签名上传链接。"""
    with open(local_path, "rb") as f:
        # 注意：上传时无须设置 Content-Type 头
        resp = requests.put(upload_url, data=f, timeout=300)
    return resp.status_code == 200


# ── 3. 轮询批量解析结果 ─────────────────────────────
def poll_batch_results(
    batch_id: str,
    interval: int = 10,
    timeout: int = 1800,
) -> list[dict]:
    """
    轮询查询批量解析结果，直到所有文件处理完成或超时。

    :param batch_id: 批量任务 ID
    :param interval: 轮询间隔（秒）
    :param timeout: 最大等待时间（秒）
    :return: 每个文件的解析结果列表，包含 full_zip_url 等字段
    """
    url = f"{BASE_URL}/extract-results/batch/{batch_id}"
    start = time.time()

    while True:
        if time.time() - start > timeout:
            raise TimeoutError(f"轮询超时（{timeout}s），batch_id={batch_id}")

        resp = requests.get(url, headers=_headers(), timeout=30)
        resp.raise_for_status()
        result = resp.json()

        if result.get("code") != 0:
            raise RuntimeError(f"查询结果失败: {result.get('msg')}")

        extract_results = result["data"]["extract_result"]

        # 检查是否全部完成（done 或 failed）
        all_finished = all(r["state"] in ("done", "failed") for r in extract_results)
        if all_finished:
            return extract_results

        done_count = sum(1 for r in extract_results if r["state"] == "done")
        print(
            f"  [轮询] {done_count}/{len(extract_results)} 已完成 (等待 {interval}s...)"
        )
        time.sleep(interval)


# ── 4. 下载结果压缩包 ──────────────────────────────
def download_result(
    full_zip_url: str,
    save_dir: str,
    file_name: str,
    data_id: str,
) -> str:
    """下载解析结果 ZIP 到本地。"""
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{file_name}_{data_id}.zip")

    resp = requests.get(full_zip_url, stream=True, timeout=300)
    resp.raise_for_status()

    with open(save_path, "wb") as f:
        f.writelines(resp.iter_content(chunk_size=8192))

    return save_path


# ── 顶层便捷函数 ─────────────────────────────────────────
def batch_upload_and_get_download_urls(
    pdf_paths: list[str],
    model_version: str = "vlm",
    enable_formula: bool = True,
    enable_table: bool = True,
    language: str = "ch",
    poll_interval: int = 10,
    poll_timeout: int = 1800,
    save_dir: str | None = None,
) -> list[dict]:
    """
    批量上传本地 PDF 并获取解析结果下载链接。

    从环境变量 MINERU_API_KEY 读取 Token。

    :param pdf_paths: 本地 PDF 文件路径列表（单次最多 50 个）
    :param model_version: pipeline / vlm / MinerU-HTML
    :param save_dir: 若提供，则自动下载 ZIP 到该目录
    :return: [{"file_name": str, "data_id": str, "state": str, "full_zip_url": str}, ...]
    """
    # 校验文件存在性
    for p in pdf_paths:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"文件不存在: {p}")

    if len(pdf_paths) > 50:
        raise ValueError(f"单次最多上传 50 个文件，当前: {len(pdf_paths)}")

    # 构建文件信息列表
    files_info = [
        {"name": os.path.basename(p), "data_id": f"file_{i}"}
        for i, p in enumerate(pdf_paths)
    ]

    # Step 1: 申请上传链接
    print(f"[1/3] 申请上传链接（{len(pdf_paths)} 个文件）...")
    upload_info = get_upload_urls(
        files=files_info,
        model_version=model_version,
        enable_formula=enable_formula,
        enable_table=enable_table,
        language=language,
    )
    batch_id = upload_info["batch_id"]
    upload_urls = upload_info["file_urls"]
    print(f"  batch_id = {batch_id}")

    # Step 2: 逐个上传
    print("[2/3] 上传文件...")
    for upload_url, local_path in zip(upload_urls, pdf_paths):
        ok = upload_file(upload_url, local_path)
        status = "✅" if ok else "❌"
        print(f"  {status} {os.path.basename(local_path)}")
        if not ok:
            raise RuntimeError(f"上传失败: {local_path}")

    # Step 3: 轮询结果
    print(f"[3/3] 等待解析完成（轮询间隔 {poll_interval}s）...")
    results = poll_batch_results(batch_id, interval=poll_interval, timeout=poll_timeout)

    # 整理返回结果 & 可选下载
    output = []
    for r in results:
        item = {
            "file_name": r["file_name"],
            "data_id": r["data_id"],
            "state": r["state"],
            "full_zip_url": r.get("full_zip_url", ""),
            "err_msg": r.get("err_msg", ""),
        }
        output.append(item)

        if save_dir and r["state"] == "done" and r.get("full_zip_url"):
            saved = download_result(
                r["full_zip_url"], save_dir, r["file_name"], r["data_id"]
            )
            print(f"  📦 已下载: {saved}")

    return output


# ── 使用示例 ────────────────────────────────────────────
if __name__ == "__main__":
    pdfs = [os.path.join(os.getcwd(), "assets", "sample.pdf")]

    results = batch_upload_and_get_download_urls(
        pdf_paths=pdfs,
        model_version="vlm",  # 推荐使用 vlm 模型
        save_dir=os.path.join(os.getcwd(), "assets"),
    )

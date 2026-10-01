import numpy as np
import torch
from FlagEmbedding import BGEM3FlagModel
from langchain_core.documents.base import Document


# 加载嵌入模型
def load_bge_m3(model_name_or_path: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model = BGEM3FlagModel(
            model_name_or_path=model_name_or_path,
            use_fp16=(device == "cuda"),
        )
        if device == "cuda":
            model.model.to(device)
        print("BGE-M3 加载完成")
        return model
    except Exception as e:  # noqa: BLE001
        print(e)


# 向量化
def get_embeddings(
    inputs: list[Document] | list[str],
    model,
    batch_size: int = 100,
) -> list[dict]:
    # 统一取出文本，保留原始对象用于后面组装 metadata
    texts = [item if isinstance(item, str) else item.page_content for item in inputs]

    total = len(texts)
    dense_vectors: list = []
    sparse_vectors: list = []

    # 分批处理
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_texts = texts[start:end]
        output = model.encode(
            sentences=batch_texts,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,  # 不需要 ColBERT，省算力
        )
        dense_vectors.extend(output["dense_vecs"])
        sparse_vectors.extend(output["lexical_weights"])

        percent = end / total * 100
        print(f"向量化进度：{percent:6.2f}% ({end}/{total})")

    # 组装最终结果
    embeddings = []
    for dense_vector, sparse_vector, input_item, text in zip(
        dense_vectors, sparse_vectors, inputs, texts
    ):
        # 存入 milvus 的稠密向量必须是 float32
        dense_vector = np.asarray(dense_vector, dtype=np.float32)
        # 存入 milvus 的稀疏向量键必须是 int
        sparse_vector = {int(k): float(v) for k, v in sparse_vector.items()}

        embedding_item = {
            "dense_vector": dense_vector,
            "sparse_vector": sparse_vector,
            "text": text,
        }
        # 如果原始输入是 Document，把 metadata 也带上
        if not isinstance(input_item, str):
            embedding_item["metadata"] = input_item.metadata
        embeddings.append(embedding_item)

    print(f"向量化完成，共 {len(embeddings)} 条")
    return embeddings

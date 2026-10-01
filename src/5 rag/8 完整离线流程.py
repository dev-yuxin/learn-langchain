import os

import numpy as np
import torch
from dotenv import load_dotenv
from FlagEmbedding import BGEM3FlagModel
from langchain_core.documents.base import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_unstructured.document_loaders import UnstructuredLoader
from pymilvus import DataType, MilvusClient

load_dotenv(override=True)


# 加载文档
def load_doc(file_path: str, mode="single"):
    try:
        loader = UnstructuredLoader(
            file_path=file_path,
            # mode 取值 "single" / "elements"
            # single: 返回列表里只有一整个文档。
            # elements: markdown 标题、段落、列表等都视为元素。word 一行视为一个元素。返回多个文档组成的列表。
            mode=mode,
        )
        docs = loader.load()
        print(f"加载{file_path}成功")
        return docs
    except Exception as e:  # noqa: BLE001
        print(e)


# 切分文档
def split_docs(docs: list[Document]):
    # 递归字符切分器
    # 对每个 doc 按分隔符优先级递归切分，直到每块不超过最大长度
    splitter = RecursiveCharacterTextSplitter(
        # 分隔符
        separators=[
            "\r\n\r\n",
            "\n\n",
            "\r\n",
            "\n",
            "。",
            "！",
            "？",
            "；",
            "：",
            "，",
            "、",
            ". ",
            "! ",
            "? ",
            "; ",
            ": ",
            ", ",
            ".",
            "!",
            "?",
            ";",
            ":",
            ",",
            " ",
            "",
        ],
        # 每块最大长度
        chunk_size=1024,
        # 相邻块重叠长度，保持语义连贯
        chunk_overlap=256,
        # 按字符数计算
        length_function=len,
    )
    chunks = splitter.split_documents(docs)
    print(f"切分为{len(chunks)}块")
    return chunks


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


# 连接数据库
def connect_milvus():
    try:
        uri = os.getenv("MILVUS_URI")
        client = MilvusClient(uri)
        print("连接数据库成功")
        return client
    except Exception as e:  # noqa: BLE001
        print(e)


# 创建 schema（对应关系数据库表的结构）
def build_schema():
    schema = MilvusClient.create_schema(auto_id=True)
    # 主键
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
    # 字段名要和向量化后的结果一一对应
    # 稠密向量
    schema.add_field(
        field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=1024
    )
    # 稀疏向量
    schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
    # 原始文本，max_length 是字节
    schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=10000)
    # 元数据
    schema.add_field(field_name="metadata", datatype=DataType.JSON)
    return schema


# 配置索引（对应关系数据库的索引）
def build_index():
    index_params = MilvusClient.prepare_index_params()
    # 稠密向量，一般用 HNSW（分层导航小世界），距离度量一般用COSINE，如果归一化了可以用IP
    # HNSW：https://milvus.io/docs/zh/v2.6.x/hnsw.md
    index_params.add_index(
        field_name="dense_vector", index_type="HNSW", metric_type="COSINE"
    )
    # 稀疏向量，只能用（SPARSE_INVERTED_INDEX）稀疏倒排索引，距离度量一般用IP（内积）
    # SPARSE_INVERTED_INDEX：https://milvus.io/docs/zh/v2.6.x/sparse-inverted-index.md
    index_params.add_index(
        field_name="sparse_vector", index_type="SPARSE_INVERTED_INDEX", metric_type="IP"
    )
    return index_params


# 创建集合（对应关系数据库的表）
def create_collection(client: MilvusClient, collection_name: str):
    try:
        if not client.has_collection(collection_name):
            client.create_collection(
                collection_name=collection_name,
                schema=build_schema(),
                index_params=build_index(),
            )
            print(f"创建集合{collection_name}成功")
        else:
            print(f"集合{collection_name}已存在")
    except Exception as e:  # noqa: BLE001
        print(e)


# 添加数据
def insert_data(
    client: MilvusClient, collection_name: str, data: list[dict], batch_size: int = 100
):
    try:
        total = len(data)
        for i in range(0, total, batch_size):
            batch = data[i : i + batch_size]
            client.insert(collection_name, batch)
            inserted = min(i + batch_size, total)
            percent = inserted / total * 100
            print(f"插入进度：{percent:.2f}% ({inserted}/{total})")
    except Exception as e:  # noqa: BLE001
        print(e)


# 完整离线流程
def main():
    # 加载文档
    docs = load_doc(os.path.join(os.getcwd(), "assets", "sample.docx"))
    if not docs:
        return
    # 切分文档
    chunks = split_docs(docs)
    if not chunks:
        return
    # 向量化
    bge_m3 = load_bge_m3(os.path.join(os.getcwd(), "models", "bge-m3"))
    if not bge_m3:
        return
    embedding_data = get_embeddings(chunks, bge_m3)
    # 写入数据库
    client = connect_milvus()
    if not client:
        return
    if client.has_collection("demo"):
        client.drop_collection("demo")
    create_collection(client, "demo")
    insert_data(client, "demo", embedding_data)


if __name__ == "__main__":
    main()

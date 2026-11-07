import os
import numpy as np
import torch
from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from FlagEmbedding import BGEM3FlagModel
from FlagEmbedding.inference.embedder.encoder_only.m3 import M3Embedder
from pymilvus import MilvusClient, DataType

load_dotenv(override=True)


def load_file(file_path: str, mode="single"):
    """加载文件（只支持md和word文档）"""
    if not os.path.isfile(file_path):
        print(f"{file_path} 文件路径错误")
        return
    suffix = os.path.splitext(file_path)[-1]
    if suffix not in [".md", ".doc", ".docx"]:
        print(f"{file_path} 不支持该文件类型")
        return
    loader = UnstructuredMarkdownLoader(file_path, mode) if suffix == ".md" else \
        UnstructuredWordDocumentLoader(file_path, mode)
    docs = loader.load()
    print(f"{file_path} 文件加载成功")
    return docs


def split_docs(docs: list[Document], chunk_size=512, chunk_overlap=64):
    """文档切分"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n# ",
            "\n\n",
            "\n",
            "。",
            "！",
            "？",
            "；",
            ". ",
            "! ",
            "? ",
            "; ",
            "，",
            ", ",
            " ",
            "",
        ],
    )
    return splitter.split_documents(docs)


def load_embedding_model():
    """使用bge-m3作为嵌入模型"""
    model = BGEM3FlagModel(
        model_name_or_path="models/bge-m3",
    )
    if torch.cuda.is_available():
        model.model = model.model.to("cuda:0")
    print(f"加载bge-m3模型成功，运行设备{next(model.model.parameters()).device}")
    return model


def docs_to_vectors(model: M3Embedder, docs: list[Document], batch_size=32):
    result = model.encode(
        sentences=[doc.page_content for doc in docs],
        return_dense=True,
        return_sparse=True,
        batch_size=batch_size
    )
    dense_vectors = result["dense_vecs"]
    sparse_vectors = result["lexical_weights"]
    print("向量化完成")
    return [{
        # 稠密向量数据类型必须是 float32
        "dense_vector": np.asarray(dense_vector, dtype=np.float32),
        # 稀疏向量键转为整形
        "sparse_vector": {int(k): v for k, v in sparse_vector.items()},
        "metadata": doc.metadata,
        "text": doc.page_content
    } for dense_vector, sparse_vector, doc in zip(dense_vectors, sparse_vectors, docs)]


def get_milvus_client():
    """连接milvus数据库"""
    client = MilvusClient(uri=os.getenv("milvus_uri"))
    print("连接数据库成功")
    return client


def create_collection(client: MilvusClient, collection_name: str):
    """创建集合"""
    if not client.has_collection(collection_name):
        # 设置集合结构
        # 设置主键自增（添加数据时无需为主键赋值）
        schema = client.create_schema(auto_id=True)
        # 主键
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
        # 原始文本，注意 max_length 是最大字节数
        schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)
        # 元数据
        schema.add_field(field_name="metadata", datatype=DataType.JSON)
        # 稠密向量（维度和你用的嵌入模型一致）
        schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=1024)
        # 稀疏向量
        schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)

        # 设置索引
        index_params = client.prepare_index_params()
        # 稠密和稀疏向量需设置索引
        # index_type 索引类型：一般稠密向量用 HNSW（分层导航小世界），稀疏向量用SPARSE_INVERTED_INDEX（稀疏倒排索引）
        # metric_type 计算相似度方式：COSINE,L2,IP，稠密向量这3种都可以，稀疏向量则只能是IP
        index_params.add_index(field_name="dense_vector", index_type="HNSW", metric_type="COSINE")
        index_params.add_index(field_name="sparse_vector", index_type="SPARSE_INVERTED_INDEX", metric_type="IP")

        client.create_collection(collection_name=collection_name, schema=schema, index_params=index_params)
        print(f"创建集合{collection_name}成功")
    else:
        print(f"集合{collection_name}已存在")


def insert_data(client: MilvusClient, collection_name: str, data: list[dict], batch_size=128):
    """添加数据"""
    total_inserted = 0
    for start in range(0, len(data), batch_size):
        batch = data[start:start + batch_size]
        if not batch:
            continue
        result = client.insert(collection_name=collection_name, data=batch)
        total_inserted += result["insert_count"]
    print(f"插入{total_inserted}条记录")
    return total_inserted


def save_rag(file_path: str, model: M3Embedder, client: MilvusClient, collection_name: str):
    docs = load_file(file_path)
    chunks = split_docs(docs)
    data = docs_to_vectors(model, chunks)
    insert_data(client, collection_name, data)


if __name__ == '__main__':
    model = load_embedding_model()
    client = get_milvus_client()
    collection_name = "test"
    create_collection(client, collection_name)
    save_rag("docs/中国科学院国家天文台2023年部门预算/full.md", model, client, collection_name)
    save_rag("docs/中华人民共和国民法典.docx", model, client, collection_name)
    save_rag("docs/投机解码.md", model, client, collection_name)

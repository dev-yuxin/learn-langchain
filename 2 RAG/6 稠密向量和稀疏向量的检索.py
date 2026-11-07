import os
import numpy as np
import torch
from dotenv import load_dotenv
from FlagEmbedding import BGEM3FlagModel
from FlagEmbedding.inference.embedder.encoder_only.m3 import M3Embedder
from pymilvus import MilvusClient

load_dotenv(override=True)


def load_embedding_model():
    """使用bge-m3作为嵌入模型"""
    model = BGEM3FlagModel(
        model_name_or_path="models/bge-m3",
    )
    if torch.cuda.is_available():
        model.model = model.model.to("cuda:0")
    print(f"加载bge-m3模型成功，运行设备{next(model.model.parameters()).device}")
    return model


def query_to_vector(query: str, model: M3Embedder):
    result = model.encode(
        sentences=[query],
        return_dense=True,
        return_sparse=True
    )
    # 稠密向量
    dense_vector = result["dense_vecs"][0]
    dense_vector = np.asarray(dense_vector, dtype=np.float32)
    # 稀疏向量
    sparse_vector = result["lexical_weights"][0]
    sparse_vector = {int(k): v for k, v in sparse_vector.items()}
    return dense_vector, sparse_vector


def get_milvus_client():
    """连接milvus数据库"""
    client = MilvusClient(uri=os.getenv("milvus_uri"))
    print("连接数据库成功")
    return client


def search_by_dense(query: str, model: M3Embedder, client: MilvusClient, collection_name: str, limit: int):
    dense_vector, _ = query_to_vector(query, model)
    return client.search(
        collection_name=collection_name,
        data=[dense_vector],
        # 对应的字段（必须和创建Collection时一致）
        anns_field="dense_vector",
        # 对应的距离度量（必须和创建Collection时一致）
        search_params={"metric_type": "COSINE"},
        # 返回几条
        limit=limit,
        # 返回的字段（一般是标量字段）
        output_fields=["text", "metadata"]
    )


def search_by_sparse(query: str, model: M3Embedder, client: MilvusClient, collection_name: str, limit: int):
    _, sparse_vector = query_to_vector(query, model)
    return client.search(
        collection_name=collection_name,
        data=[sparse_vector],
        # 对应的字段（必须和创建Collection时一致）
        anns_field="sparse_vector",
        # 对应的距离度量（必须和创建Collection时一致）
        search_params={"metric_type": "IP"},
        # 返回几条
        limit=limit,
        # 返回的字段（一般是标量字段）
        output_fields=["text", "metadata"]
    )


if __name__ == '__main__':
    model = load_embedding_model()
    client = get_milvus_client()

    result = search_by_dense("自然人下落不明几年可视为死亡", model, client, "test", limit=5)
    # result[0]对应第几个查询向量
    for item in result[0]:
        # item 是一个类似字典的对象，对应该查询向量的检索结果，一共 limit 条
        # distance是粉丝
        print(item["distance"])
        # output_fields 会放入 entity 中
        print(item["entity"]["text"])
        print(item["entity"]["metadata"])
        print("=" * 100)

    result = search_by_sparse("自然人下落不明几年可视为死亡", model, client, "test", limit=3)
    for item in result[0]:
        print(item["distance"])
        print(item["entity"]["text"])
        print(item["entity"]["metadata"])
        print("=" * 100)

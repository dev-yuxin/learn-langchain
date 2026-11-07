import os
import numpy as np
import torch
from dotenv import load_dotenv
from FlagEmbedding import BGEM3FlagModel
from FlagEmbedding.inference.embedder.encoder_only.m3 import M3Embedder
from pymilvus import MilvusClient, AnnSearchRequest, RRFRanker, WeightedRanker

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


def hybrid_search(
        query: str,
        model: M3Embedder,
        client: MilvusClient,
        collection_name: str,
        dense_limit: int,
        sparse_limit: int,
        limit: int
):
    dense_vector, sparse_vector = query_to_vector(query, model)
    # 稠密检索
    req_dense = AnnSearchRequest(
        data=[dense_vector],
        anns_field="dense_vector",
        param={"metric_type": "COSINE"},
        limit=dense_limit
    )
    # 稀疏检索
    req_sparse = AnnSearchRequest(
        data=[sparse_vector],
        anns_field="sparse_vector",
        param={"metric_type": "IP"},
        limit=sparse_limit
    )
    # 混合检索
    result = client.hybrid_search(
        collection_name=collection_name,
        reqs=[req_dense, req_sparse],
        # 重排序
        # RRFRanker: 不看distance，只看排名，基于排名的倒数排序（一般用于同源的数据）
        # WeightedRanker: 看distance，加权排名（一般非同源数据且知道权重）
        ranker=RRFRanker(k=60),  # k是平滑常数 1/(k+排名)
        limit=limit,
        output_fields=["text", "metadata"]
    )
    context = ""
    for item in result[0]:
        context += "\n" + "=" * 100 + "\n"
        context += item["entity"]["text"]
        context += "\n" + "=" * 100 + "\n"
    return context


if __name__ == '__main__':
    model = load_embedding_model()
    client = get_milvus_client()
    context = hybrid_search("自然人下落不明几年可视为死亡", model, client, "test", 5, 3, 3)
    print(context)

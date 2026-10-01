import os

import numpy as np
import torch
from dotenv import load_dotenv
from FlagEmbedding import BGEM3FlagModel
from langchain.chat_models import init_chat_model
from langchain_core.documents.base import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from pymilvus import AnnSearchRequest, MilvusClient, RRFRanker

load_dotenv(override=True)


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


# 混合检索
def hybrid_search(
    client: MilvusClient,
    collection_name: str,
    dense_data: list[list],
    sparse_data: list[list],
    dense_limit=10,
    sparse_limit=10,
    all_limit=5,
):
    dense_search = AnnSearchRequest(
        data=dense_data,
        anns_field="dense_vector",
        param={"metric_type": "COSINE"},
        limit=dense_limit,
    )
    sparse_search = AnnSearchRequest(
        data=sparse_data,
        anns_field="sparse_vector",
        param={"metric_type": "IP"},
        limit=sparse_limit,
    )
    return client.hybrid_search(
        collection_name=collection_name,
        reqs=[dense_search, sparse_search],
        # 这里使用 RRFRanker 基于排名倒数的重排序
        # RRFRanker: https://milvus.io/docs/zh/v2.6.x/rrf-ranker.md#RRF-Ranker
        # 如果你想指定权重你也可以使用 WeightedRanker 基于权重的重排序
        # WeightedRanker: https://milvus.io/docs/zh/v2.6.x/reranking.md#WeightedRanker
        ranker=RRFRanker(),
        limit=all_limit,
        output_fields=["id", "text", "metadata"],
    )


def retriever(query: str):
    try:
        bge_m3 = load_bge_m3(os.path.join(os.getcwd(), "models", "bge-m3"))
        embeddings = get_embeddings([query], bge_m3)
        dense_vector = embeddings[0]["dense_vector"]
        sparse_vector = embeddings[0]["sparse_vector"]
        client = connect_milvus()
        result = hybrid_search(
            client=client,
            collection_name="demo",
            dense_data=[dense_vector],
            sparse_data=[sparse_vector],
        )
        context = ""
        for item in result[0]:
            text = item["entity"]["text"]
            metadata = item["entity"]["metadata"]
            context += f"""\n{text}
            来源：{metadata}\n
            """
        print("=" * 100)
        print(f"检索到上下文\n{context}")
        print("=" * 100)
        return context
    except Exception as e:  # noqa: BLE001
        print(e)
        return ""


if __name__ == "__main__":
    chain = (
        {"query": RunnablePassthrough(), "context": retriever}
        | ChatPromptTemplate.from_messages(
            [
                ("system", "你是一个民法典相关的助手，你要回答用户民法典相关问题。"),
                ("human", "请根据上下文背景{context}回答提问：{query}"),
            ]
        )
        | init_chat_model(model="deepseek:deepseek-flash")
        | StrOutputParser()
    )
    answer = chain.invoke("自然人什么情况下视为失踪")
    print(answer)

import os
from dotenv import load_dotenv
from pymilvus import MilvusClient, AnnSearchRequest, RRFRanker, WeightedRanker

load_dotenv(override=True)


def get_milvus_client():
    """连接milvus数据库"""
    client = MilvusClient(uri=os.getenv("milvus_uri"))
    print("连接数据库成功")
    return client


def query(client: MilvusClient, collection_name: str, keyword: str):
    return client.query(
        collection_name=collection_name,
        filter=f"text like '%{keyword}%'",
        output_fields=["text", "metadata"]
    )


if __name__ == '__main__':
    client = get_milvus_client()
    result = query(client, "test", "预算")
    for item in result:
        print(item["text"])
        print(item["metadata"])
        print("=" * 100)

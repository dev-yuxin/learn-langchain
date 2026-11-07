import os

from dotenv import load_dotenv
from pyarrow.lib import DataType
from pymilvus import MilvusClient, DataType

load_dotenv(override=True)


# 连接 milvus
def get_milvus_client():
    client = MilvusClient(uri=os.getenv("milvus_uri"))
    return client


# 创建集合
def create_collection(client: MilvusClient, collection_name: str):
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


# 添加数据
def insert_data(
        client: MilvusClient,
        collection_name: str,
        data: list[dict],
        batch_size=128,
) -> int:
    total_inserted = 0
    for start in range(0, len(data), batch_size):
        batch = data[start:start + batch_size]
        if not batch:
            continue
        result = client.insert(collection_name=collection_name, data=batch)
        total_inserted += result["insert_count"]
    print(f"插入{total_inserted}条记录")
    return total_inserted


if __name__ == '__main__':
    client = get_milvus_client()
    create_collection(client, "test")

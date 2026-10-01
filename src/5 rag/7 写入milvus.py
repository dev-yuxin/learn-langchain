import os

from dotenv import load_dotenv
from pymilvus import DataType, MilvusClient

load_dotenv(override=True)


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

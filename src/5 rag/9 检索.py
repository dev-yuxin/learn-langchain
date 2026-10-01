from pymilvus import AnnSearchRequest, MilvusClient, RRFRanker


# 根据稠密向量搜索
def search_by_dense_vector(
    client: MilvusClient,
    collection_name: str,
    dense_data: list[list],
    limit=10,
):
    return client.search(
        data=dense_data,
        collection_name=collection_name,
        anns_field="dense_vector",
        search_params={"metric_type": "COSINE"},
        output_fields=["id", "text", "metadata"],
        limit=limit,
    )


# 根据稀疏向量搜索
def search_by_sparse_vector(
    client: MilvusClient,
    collection_name: str,
    sparse_data: list[dict],
    limit=10,
):
    return client.search(
        data=sparse_data,
        collection_name=collection_name,
        anns_field="sparse_vector",
        search_params={"metric_type": "IP"},
        output_fields=["id", "text", "metadata"],
        limit=limit,
    )


# 混合检索
def hybrid_search(
    client: MilvusClient,
    collection_name: str,
    dense_data: list[list],
    sparse_data: list[dict],
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


# 标量检索
# 类似 sql ，例如 filter="text like '%合同%'"
def query(client: MilvusClient, collection_name: str, filter: str, limit=5):
    return client.query(
        collection_name=collection_name,
        filter=filter,
        output_fields=["id", "text", "metadata"],
        limit=limit,
    )

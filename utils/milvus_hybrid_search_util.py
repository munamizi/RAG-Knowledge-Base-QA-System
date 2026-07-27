from pymilvus import AnnSearchRequest, WeightedRanker

from utils.logger import logger


def create_hybrid_search_requests(
        dense_vector,
        sparse_vector,
        dense_params=None,
        sparse_params=None,
        expr=None,
        limit=5
):
    """
    构建Milvus混合搜索请求
    Args:
        dense_vector:  文本稠密向量
        sparse_vector: 文本稀疏向量
        dense_params:  稠密向量度量类型，默认使用余弦相似度
        sparse_params: 稀疏向量度量类型，默认使用内点积
        expr: 过滤表达式，用于精准筛选数据
        limit: 单向量搜索返回结果数量，默认5
    Returns:
        搜索请求列表，包含[dense_req, sparse_req]
    """
    if dense_params is None:
        dense_params = {"metric_type": "COSINE"}
    if sparse_params is None:
        sparse_params = {"metric_type": "IP"}

    dense_req = AnnSearchRequest(
        data=[dense_vector],
        anns_field="dense_vector",
        param=dense_params,
        expr=expr,
        limit=limit
    )
    sparse_req = AnnSearchRequest(
        data=[sparse_vector],
        anns_field="sparse_vector",
        param=sparse_params,
        expr=expr,
        limit=limit
    )

    return [dense_req, sparse_req]




def hybrid_search(
        client,
        collection_name,
        reqs,
        ranker_weights=(0.5, 0.5),
        norm_score=False,
        limit=5,
        output_fields=None,
        search_params=None
):
    """
    执行Milvus混合搜索

    Args:
        client: MilvusClient实例
        collection_name: 集合名称
        reqs: 搜索请求列表，固定为[dense_req, sparse_req]
        ranker_weights: 加权融合权重，默认(0.5,0.5)，依次对应稠密/稀疏向量
        norm_score: 是否归一化评分后再融合，避免评分量级差异导致权重失效
        limit: 混合搜索最终返回结果数量，默认5
        output_fields: 需要返回的字段列表，默认返回item_name
        search_params: 搜索参数，如ef/topk等，默认None

    Returns:
        混合搜索结果列表，搜索失败返回None
    """
    try:
        # 初始化加权排名器：按权重融合稠密/稀疏向量的搜索结果
        # norm_score=True：先将两个向量评分归一化到0~1区间，再加权计算，避免一个得分特别大、另一个特别小导致权重失效。
        rerank = WeightedRanker(ranker_weights[0], ranker_weights[1], norm_score=norm_score)
        # 默认返回字段：文档标识字段
        if output_fields is None:
            output_fields = ["item_name"]

        # 执行混合搜索：融合稠密+稀疏向量结果，按权重重新排序
        res = client.hybrid_search(
            collection_name=collection_name,
            reqs=reqs,
            ranker=rerank,
            limit=limit,
            output_fields=output_fields,
            search_params=search_params
        )

        logger.info(f"Milvus混合搜索完成，集合[{collection_name}]共检索到{len(res[0])}条结果")
        return res
    except Exception as e:
        logger.error(f"Milvus混合搜索执行失败，集合[{collection_name}]：{str(e)}", exc_info=True)
        return None
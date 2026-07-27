from processor.config_plus.milvus_config import milvus_config
from processor.query_processor.BaseNode import BaseNode
from processor.query_processor.state import QueryGraphState
from utils.embedding_util import generate_embeddings
from utils.logger import logger
from utils.milvus_client_util import get_milvus_client
from utils.milvus_hybrid_search_util import create_hybrid_search_requests, hybrid_search


class NodeSearchEmbedding(BaseNode):
    """
   基于已确认主体名+改写后的用户问题，执行Milvus向量数据库混合检索
   """


    name: str = "node_search_embedding"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        """
        核心节点函数：基于已确认商品名+改写后的用户问题，执行Milvus向量数据库混合检索
        Args:
            state - 核心字段：
                          {
                              "rewritten_query": str,   # step4改写后的完整用户问题
                              "item_names": list[str],  # step7已确认的标准化商品名列表
                          }

        Returns:
            Dict - 检索结果字典：
                         {
                             "embedding_chunks": List[Dict]  # Milvus检索结果列表，无结果则为空列表
                                                             # 每个元素为一条匹配的向量数据，含业务字段
                         }
        """

        try:

            # 1、用户问题和已确认商品名
            query = state.get("rewritten_query")
            item_names = state.get("item_names")

            # 2、向量化用户问题，得稠密+稀疏向量
            embeddings = generate_embeddings([query])
            dense_vec = embeddings.get("dense")[0]
            sparse_vec = embeddings.get("sparse")[0]

            # 3. 获取Milvus集合名
            collection_name = milvus_config.chunks_collection

            # 4、处理 item_names 中的引号，防止注入或语法错误
            expr = None
            if item_names:
                expr = f'item_name in {item_names}'
                logger.info(f"过滤条件: {expr}")
            else:
                logger.info("未指定商品名过滤，将全库检索")

            # 5、构造混合检索请求对象
            reqs = create_hybrid_search_requests(
                dense_vector=dense_vec,
                sparse_vector=sparse_vec,
                expr=expr,
                limit=5
            )

            # 6、执行混合检索
            logger.info("开始执行 Milvus 混合检索...")
            client = get_milvus_client()
            res = hybrid_search(
                client=client,
                collection_name=collection_name,
                reqs=reqs,
                ranker_weights=(0.8, 0.2),
                output_fields=["chunk_id", "content", "item_name"]
            )

            # 7、构造并返回结果：若检索结果非空，取res[0]，否则返回空列表
            return {"embedding_chunks": res[0] if res else []}

        except Exception as e:
            logger.exception(f"向量搜索失败: {e}")





if __name__ == "__main__":

    init_state = {
        "rewritten_query": "关于brother HAK180烫金机，如何调节转印温度？",
        "item_names": ["BrotherHAK180烫金机", "BrotherHAK-180烫金机"]
    }
    node_search_embedding = NodeSearchEmbedding()
    result = node_search_embedding(init_state)

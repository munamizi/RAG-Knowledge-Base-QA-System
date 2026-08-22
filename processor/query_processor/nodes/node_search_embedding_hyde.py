from langchain_core.output_parsers import StrOutputParser

from processor.configs.milvus_config import milvus_config
from processor.query_processor.BaseNode import BaseNode
from processor.query_processor.prompt.search_embedding_hyde import HYDE_PROMPT
from processor.query_processor.state import QueryGraphState
from utils.embedding_util import generate_embeddings
from utils.llm_util import get_llm_client
from utils.logger import logger
from utils.milvus_client_util import get_milvus_client
from utils.milvus_hybrid_search_util import create_hybrid_search_requests, hybrid_search


class NodeSearchEmbeddingHyde(BaseNode):

    name: str = "node_search_embedding_hyde"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        """
        HyDE检索节点
        内容：
            1. 参数提取：state[rewritten_query]、state[item_names]
            2. 生成假设文档：调用LLM，基于用户问题生成一段假设性回答。
            3. 混合检索：
                将 "用户问题 + 假设文档"合并，BGE-M3生成稠密+稀疏向量。
                在Milvus中执行混合检索（带item_names商品名过滤）。
            4. 结果封装：返回检索到的切片列表和生成的假设文档，更新会话状态。

        Args:
            state[rewritten_query] - 改写后的问题、
            state[item_names] - 提取的商品名

        Returns:
            state[hyde_embedding_chunks] - 检索结果
            state[hyde_doc] - 假设文档的字典

        """

        # 1.参数提取
        rewritten_query = state.get("rewritten_query")
        item_names = state.get("item_names")

        try:

            # 2.生成假设性文档
            hyde_doc = self._step_1_create_hyde_doc(rewritten_query)

            # 3.混合检索
            res = self._step_2_search_embedding_hyde(
                rewritten_query=rewritten_query,
                hyde_doc=hyde_doc,
                item_names=item_names
            )

            logger.info(res)
            # 4.结果封装
            return {
                "hyde_embedding_chunks": res,
                "hyde_doc": hyde_doc,
            }

        except Exception as e:
            logger.exception(f"假设性文档向量搜索失败: {e}")



    def _step_1_create_hyde_doc(self, rewritten_query: str) -> str:
        """
        利用LLM根据用户查询生成假设性文档

        Args:
            rewritten_query: 用户改写后的查询语句

        Returns:
            hyde_doc: LLM生成的假设性文档内容

        """

        logger.info("步骤1: 开始生成假设性文档")

        try:
            llm = get_llm_client()
            hyde_prompt = HYDE_PROMPT.format(rewritten_query=rewritten_query)
            hyde_doc = StrOutputParser().invoke(llm.invoke(hyde_prompt))
            return hyde_doc

        except Exception as e:
            logger.exception(f"步骤1: 生成假设文档失败: {e}")
            raise e



    def _step_2_search_embedding_hyde(
            self,
            rewritten_query: str,
            hyde_doc: str,
            item_names=None
    ):
        """
        将 "重写问题rewritten_query"+ "假设性文档HyDE_doc"合并为整体，并到向量库检索切片。

        Args:
            rewritten_query: str - 改写后的查询
            hyde_doc: str - 生成的假设性文档
            item_names: list[str] - 商品名称列表，用于表达式限定检索范围

        Returns:
            res: list[dict] - 检索结果列表
        """

        try:
            combined_text = rewritten_query + " " + hyde_doc

            embeddings = generate_embeddings([combined_text])
            dense_vec = embeddings.get("dense")[0]
            sparse_vec = embeddings.get("sparse")[0]

            collection_name = milvus_config.chunks_collection

            expr = None
            if item_names:

                expr = f'item_name in {item_names}'
                logger.info(f"步骤2: 过滤条件: {expr}")
            else:
                logger.info("步骤2: 未指定商品名过滤，将全库检索")

            # 构造Milvus混合搜索请求，执行混合检索
            reqs = create_hybrid_search_requests(
                dense_vector=dense_vec,
                sparse_vector=sparse_vec,
                expr=expr,
                limit=5
            )

            logger.info("步骤2: 开始执行 Milvus 混合检索...")
            client = get_milvus_client()
            res = hybrid_search(
                client=client,
                collection_name=collection_name,
                reqs=reqs,
                ranker_weights=(0.8, 0.2),
                output_fields=["chunk_id", "content", "item_name"],
            )

            return res[0] if res else []

        except Exception as e:
            logger.error(f"步骤2: 检索过程发生异常: {e}")
            raise e


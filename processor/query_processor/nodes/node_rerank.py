from typing import List, Dict, Any

from fastapi import FastAPI

from processor.configs.topk_truncate import truncate_config
from processor.query_processor.BaseNode import BaseNode
from processor.query_processor.state import QueryGraphState
from utils.logger import logger
from utils.reranker_http_util import rerank_documents


class NodeRerank(BaseNode):

    name: str = "node_rerank"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        """
        使用 Cross-Encoder 模型对多路召回结果做精排，并做动态TopK

        内容:
            1.合并多源文档为格式：
            2.Reranker精排
            3.动态TopK（断崖检测截断）

        Args:
            state['rrf_chunks']
            state['web_search_docs']
            state['rewritten_query']

        Returns:
            state['reranked_docs']

        """

        # 1. 合并多源文档
        merged_multi_docs: List[Dict[str, Any]] = self._step_1_merge_multi_source_docs(state)

        # 2. Rerank模型精排
        reranked_docs: List[Dict[str, Any]] = self._step_2_rerank_merged_docs(state, merged_multi_docs)

        # 3. 动态TopK(断崖检测截断)
        cutoff_docs = self._step_3_cliff_cutoff(reranked_docs)

        # 4. 更新state
        state['reranked_docs'] = cutoff_docs
        return state



    def _step_1_merge_multi_source_docs(self, state: QueryGraphState) -> List[Dict[str, Any]]:
        """
        合并三路召回结果为统一格式

        Args:
            state['rrf_chunks']
            state['web_search_docs']

        Returns:
            merged_multi_docs - 合并后的结果：
                                        [
                                            {
                                                "content": 值
                                                "chunk_id": 值
                                                "url": 值或None,
                                                "source": "local" or "web"
                                            },
                                            ......
                                        ]

        """

        final_docs = []

        # 1. 获取本地 RRF （普通检索+HyDE检索）的文档
        for rrf_doc in state.get('rrf_chunks') or []:
            format_rrf_doc = {
                "content": rrf_doc.get('content'),
                "chunk_id": rrf_doc.get('chunk_id'),
                "url": None,
                "source": "local"
            }
            final_docs.append(format_rrf_doc)

        # 2. 获取 web search的文档
        for web_doc in state.get('web_search_docs') or []:
            format_web_doc = {
                "content": web_doc.get('snippet'),
                "chunk_id": None,
                "url": web_doc.get('url'),
                "source": "web"
            }
            final_docs.append(format_web_doc)

        return final_docs



    def _step_2_rerank_merged_docs(self, state: QueryGraphState, merged_multi_docs: List[Dict[str, Any]]) -> List[
        Dict[str, Any]]:
        """
        使用 Reranker 模型对文档进行精排

        Args:
            state['rewritten_query']
            merged_multi_docs - 步骤一结果

        Returns:
            sorted_score_docs - 被添加score分数并降序的字典：
                                     [
                                            {
                                                "content": 值
                                                "chunk_id": 值
                                                "url": 值或None,
                                                "source": "local" or "web"
                                                "score":精排得到的分数
                                            },
                                            ......
                                        ]

        """

        if not merged_multi_docs:
            logger.warning("没有可用于 Rerank 的文档")
            return []


        try:
            user_query = state.get('rewritten_query')
            contents = [doc.get("content") for doc in merged_multi_docs]
            # 调用Rerank
            rerank_scores = rerank_documents(user_query, contents)
            scored_docs = [{**doc, "score": score} for doc, score in zip(merged_multi_docs, rerank_scores)]


            sorted_score_docs = sorted(
                scored_docs,
                key=lambda x: x["score"],
                reverse=True
            )

            return sorted_score_docs

        except Exception as e:
            logger.error(f"Rerank 重排序失败: {str(e)}")
            return []



    def _step_3_cliff_cutoff(self, ranked_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        断崖检测截断（最少返回3条，最大返回10条且满足下面任一条件就截断）：
            ①绝对差值：A-B ≥ 0.5
            ②相对差值：(B-A)/A ≥ 25%
        Args:
            ranked_docs - 步骤二结果：
                             [
                                {
                                    "content": 值
                                    "chunk_id": 值
                                    "url": 值或None,
                                    "source": "local" or "web"
                                    "score":精排得到的分数
                                },
                                ......
                            ]

        Returns:
            state['reranked_docs']
                            [
                                {
                                    "content": 值
                                    "chunk_id": 值
                                    "url": 值或None,
                                    "source": "local" or "web"
                                    "score":精排得到的分数
                                },
                                ......
                            ]
        """
        if not ranked_docs:
            return []

        upper_bound = min(truncate_config.rerank_max_topk, len(ranked_docs))
        lower_bound = min(truncate_config.rerank_min_topk, upper_bound)

        # 默认值：取满硬上限（最多10条）
        cutoff_pos = upper_bound

        for idx in range(lower_bound - 1, upper_bound - 1):
            current_score = ranked_docs[idx].get("score")
            next_score = ranked_docs[idx + 1].get("score")

            if current_score is None or next_score is None:
                continue
            # 计算绝对差值
            abs_gap = current_score - next_score
            # 计算相对差值
            rel_gap = abs_gap / (abs(current_score) + 1e-6)
            # 断崖检测
            if abs_gap >= truncate_config.rerank_gap_abs or rel_gap >= truncate_config.rerank_gap_ratio:
                # 最终取前i+1条（索引转实际数量，如i=2 → 取前3条）
                cutoff_pos = idx + 1
                logger.debug(f"断崖检测: 位置 {idx + 1}, abs_gap={abs_gap:.4f}, rel_gap={rel_gap:.4f}")
                break

        return ranked_docs[:cutoff_pos]




if __name__ == "__main__":

    mock_state = {
        "rewritten_query": "怎么测这块主板的短路问题？",
        "rrf_chunks": [
            {
                "chunk_id": "local_1",
                "content": "主板短路通常表现为通电后风扇转一下就停，可以使用万用表的蜂鸣档测量。"
            },
            {
                "chunk_id": "local_2",
                "content": "今天中午去吃猪脚饭吧，这块主板外观很漂亮。"
            },
        ],
        "web_search_docs": [
            {
                "url": "https://example.com/repair",
                "snippet": "主板通电前先打各主供电电感对地阻值，阻值偏低就是短路。"
            },
            {
                "url": "https://example.com/news",
                "snippet": "苹果发布新款手机，A系列芯片性能提升20%。"
            },
        ],
    }

    node_rerank = NodeRerank()
    result = node_rerank(mock_state)

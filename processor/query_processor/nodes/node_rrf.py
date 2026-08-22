from typing import List, Tuple, Dict, Any

from processor.query_processor.BaseNode import BaseNode
from processor.query_processor.state import QueryGraphState


class NodeRrf(BaseNode):

    name: str = "node_rrf"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        """
        将二路召回的结果（向量、HyDE）进行RRF融合排序

        Args:
            state[embedding_chunks]
            state[hyde_embedding_chunks]

        Returns:
            state[rrf_chunks] : list[Dict]
                [
                    {
                        'chunk_id':值
                        'content':值
                    },
                    ......
                ]
        """
        # 1. 获取普通检索和HyDE检索结果的entity
        embedding_search_list = [doc.get('entity') for doc in (state.get('embedding_chunks') or []) ]
        hyde_embedding_search_list = [doc.get('entity') for doc in (state.get('hyde_embedding_chunks') or [])]

        # 2. 为检索结果设置权重，并封装格式：
        rrf_inputs  = [
            (embedding_search_list, 1.0),
            (hyde_embedding_search_list, 1.0)
        ]

        # 3. 用RRF公式去获取到所有路查询到的所有chunk对应的score
        rrf_merge_results = self._rrf_merge(rrf_inputs)

        # 4. 获取rrf_chunks
        rrf_chunks = [doc for doc, _ in rrf_merge_results]

        # 5. 更新state
        state['rrf_chunks'] = rrf_chunks
        return state



    def _rrf_merge(
            self,
            rrf_inputs: list[Tuple[list[Dict], float]],
            k: int = 60,
            max_results: int = None
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        利用 RRF 公式计算每一个文档的总得分

        Args:
            rrf_inputs - 整理好的(结果字典列表，权重)的列表
            k - 平滑参数，取60
            max_results -  合并完之后返回的文档数，None 表示全部

        Returns:
            sorted_results - 合并以及排序后的文档列表(按得分降序)
            [
                (元素, 得分),
                ...
            ]

        """
        chunk_scores = {}  # 存放所有 chunk 的 RRF 计算后的分数值
        chunk_data = {}    # 存放所有 chunk 的文档数据

        for rrf_input, weight in rrf_inputs:
            for rank, doc in enumerate(rrf_input, start=1):
                chunk_id = doc.get('chunk_id')
                # RRF 公式
                chunk_scores[chunk_id] = chunk_scores.get(chunk_id, 0.0) + weight / (k + rank)

                # 使用 setdefault 保留首次遇到的文档版本(只记录第一次)
                chunk_data.setdefault(chunk_id, doc)

        # 按得分降序排序
        unsorted_results = [(chunk_data[cid], score) for cid, score in chunk_scores.items()]
        sorted_results = sorted(
            unsorted_results,
            # 排序时看每个元素的第 2 个值（也就是分数）
            key=lambda x: x[1],
            reverse=True
        )

        # 动态截取前 max_results 条
        return sorted_results[:max_results] if max_results else sorted_results




if __name__ == "__main__":
    # 模拟两路检索结果
    mock_state = {
        "embedding_chunks": [
            {
                "entity": {
                    "chunk_id": "chunk_1",
                    "content": "向量搜索结果1",
                }
            },
            {
                "entity": {
                    "chunk_id": "chunk_2",
                    "content": "向量搜索结果2",
                }
            },
            {
                "entity": {
                    "chunk_id": "chunk_3",
                    "content": "向量搜索结果3",
                }
            },
        ],
        "hyde_embedding_chunks": [
            {
                "entity": {
                    "chunk_id": "chunk_1",
                    "content": "HyDE搜索结果1",
                }
            },
            {
                "entity": {
                    "chunk_id": "chunk_4",
                    "content": "HyDE搜索结果4",
                }
            },
            {
                "entity": {
                    "chunk_id": "chunk_2",
                    "content": "HyDE搜索结果2",
                }
            },
        ],
    }

    node = NodeRrf()

    # 测试完整节点
    node(mock_state)

    print(f"RRF融合后的结果：{mock_state.get('rrf_chunks')}")


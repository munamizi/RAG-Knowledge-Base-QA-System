from typing import TypedDict, List

class QueryGraphState(TypedDict):
    """查询流程图状态"""

    session_id: str  # 会话ID
    message_id: str  # 消息ID

    original_query: str  # 用户原始问题

    # ——————————多路召回的数据——————————
    embedding_chunks: list       # 普通检索得到的[结果1，结果2...]
    hyde_embedding_chunks: list  # HyDE检索得到的[结果1，结果2...]
    web_search_docs: list        # web_search回来的文档

    # ——————————重排的数据—————————————
    rrf_chunks: list     # RRF 融合排序后的切片
    reranked_docs: list  # 重排序后的最终 Top-K 文档

    # ——————————生成的数据—————————————
    prompt: str              # 组装好的 Prompt
    answer: str              # 最终生成的答案
    reference_sources: list  # 进入本轮llm生成的编号-文本块对应表
    citations: list          # 答案实际引用的来源及内容快照
    citation_warnings: list  # 无效编号或未引用提示；不代表语义正确性判断

    # ——————————辅助——————————————————
    item_names: List[str]  # 提取的商品名
    rewritten_query: str   # 改写后的问题
    history: list          # 历史对话记录
    is_stream: bool        # 是否流式输出
    hyde_doc:str           # llm生成的假设文档
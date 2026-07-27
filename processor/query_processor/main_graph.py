from typing import Optional

from langgraph.constants import END
from langgraph.graph import StateGraph

from processor.query_processor.nodes.node_answer_output import NodeAnswerOutput
from processor.query_processor.nodes.node_item_name_confirm import NodeItemNameConfirm
from processor.query_processor.nodes.node_rerank import NodeRerank
from processor.query_processor.nodes.node_rrf import NodeRrf
from processor.query_processor.nodes.node_search_embedding import NodeSearchEmbedding
from processor.query_processor.nodes.node_search_embedding_hyde import NodeSearchEmbeddingHyde
from processor.query_processor.nodes.node_web_search_mcp import NodeWebSearchMcp
from processor.query_processor.state import QueryGraphState
from utils.logger import logger


class KBQueryWorkflow:
    """
    创建实例后自动构建图（未编译）
        run()：对图的编译和执行(invoke、stream)进行封装
        compile()：可手动编译图
    """

    def __init__(self):
        # 1. 创建图实例
        self.workflow = StateGraph(QueryGraphState)
        # 2. 创建所有业务节点
        self._init_nodes()
        # 3. 添加节点
        self._register_nodes()
        # 4. 添加边
        self._setup_routes()
        # 编译标记
        self._compiled: Optional[object] = None



    def _init_nodes(self):
        self.node_item_name_confirm = NodeItemNameConfirm()
        self.node_search_embedding = NodeSearchEmbedding()
        self.node_search_embedding_hyde = NodeSearchEmbeddingHyde()
        self.node_web_search_mcp = NodeWebSearchMcp()
        self.node_rrf = NodeRrf()
        self.node_rerank = NodeRerank()
        self.node_answer_output = NodeAnswerOutput()



    def _register_nodes(self):
        # 节点标识与实例属性名保持一致，便于维护
        self.workflow.add_node("node_item_name_confirm", self.node_item_name_confirm) # 确认主体
        self.workflow.add_node("node_search_embedding", self.node_search_embedding) # 向量搜索
        self.workflow.add_node("node_search_embedding_hyde", self.node_search_embedding_hyde) #假设性答案向量搜索
        self.workflow.add_node("node_web_search_mcp", self.node_web_search_mcp) # 联网搜索
        self.workflow.add_node("node_rrf", self.node_rrf) # 排序
        self.workflow.add_node("node_rerank", self.node_rerank) # 重排
        self.workflow.add_node("node_answer_output", self.node_answer_output) # 生成



    def _route_after_item_name_confirm(self, state: QueryGraphState) -> str|list[str]:
        if state.get("answer"):
            """
            这主要发生在 node_item_name_confirm 节点无法直接确定唯一的商品型号，从而需要“反问用户”或“拒绝回答”的场景。
            具体来说，有以下两种情况会导致 state 中直接出现 answer ，从而跳过后续的检索流程，直接输出：
            1. 多选一（反问用户） ：
            - 场景 ：用户问得太模糊（比如“华为P60”），系统发现数据库里有“华为P60 128G”和“华为P60 Art”两个型号，不足以直接确认。
            - 处理 ：节点会生成一条反问句作为 answer ，例如：“您是想问以下哪个产品：华为P60 128G、华为P60 Art？请明确一下型号。”
            - 结果 ：此时不需要去检索文档了，直接把这句话发给用户让他选。

            2. 查无此人（拒绝回答） ：
            - 场景 ：用户问了一个系统里压根没有的商品（比如“小米15”，库里只有华为的数据）。
            - 处理 ：节点会生成一条拒绝句作为 answer ，例如：“抱歉，未找到相关产品，请提供准确型号以便我为您查询。”
            - 结果 ：同样不需要后续检索，直接结束流程。
            """
            return "node_answer_output"
        else:
            # 否则继续搜索流程
            return ["node_search_embedding", "node_search_embedding_hyde", "node_web_search_mcp"]



    def _setup_routes(self):
        self.workflow.set_entry_point("node_item_name_confirm")

        self.workflow.add_conditional_edges(
            "node_item_name_confirm",
            self._route_after_item_name_confirm,
            {
                "node_answer_output": "node_answer_output",
                "node_search_embedding": "node_search_embedding",
                "node_search_embedding_hyde": "node_search_embedding_hyde",
                "node_web_search_mcp": "node_web_search_mcp"
            }
        )

        self.workflow.add_edge("node_search_embedding", "node_rrf")
        self.workflow.add_edge("node_search_embedding_hyde", "node_rrf")
        self.workflow.add_edge("node_web_search_mcp", "node_rrf")
        self.workflow.add_edge("node_rrf", "node_rerank")
        self.workflow.add_edge("node_rerank", "node_answer_output")
        self.workflow.add_edge("node_answer_output", END)



    def compile(self):
        """编译工作流（公开方法，支持手动触发编译）"""
        if not self._compiled:
            self._compiled = self.workflow.compile()
        return self._compiled



    def run(self, initial_state: QueryGraphState, stream: bool = False) -> QueryGraphState:
        """
        统一执行入口，支持切换invoke/stream
        Args:
            initial_state:  初始state
            stream: 流式输出开关
        Returns:
            state：更新后state
        """
        """"""
        if not self._compiled:
            self.compile()

        self._compiled.get_graph().print_ascii()

        if stream:
            return self._compiled.stream(initial_state)
        else:
            return self._compiled.invoke(initial_state)



if __name__ == "__main__":

    # 定义初始状态
    init_state = { "original_query": "如何使用万用表测量电压？"}

    graph = KBQueryWorkflow()
    # final_state = workflow.run(init_state)
    # logger.info(final_state)
    # 流式输出
    for chunk in graph.run(init_state):
        logger.warning(chunk)



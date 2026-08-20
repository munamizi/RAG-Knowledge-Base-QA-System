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
        # 创建图实例
        self.workflow = StateGraph(QueryGraphState)
        # 创建所有节点
        self._init_nodes()
        # 添加节点
        self._register_nodes()
        # 添加边
        self._setup_routes()
        # 编译后的图
        self._compiled_graph: Optional[object] = None



    def _init_nodes(self):
        self.node_item_name_confirm = NodeItemNameConfirm()
        self.node_search_embedding = NodeSearchEmbedding()
        self.node_search_embedding_hyde = NodeSearchEmbeddingHyde()
        self.node_web_search_mcp = NodeWebSearchMcp()
        self.node_rrf = NodeRrf()
        self.node_rerank = NodeRerank()
        self.node_answer_output = NodeAnswerOutput()



    def _register_nodes(self):
        self.workflow.add_node("node_item_name_confirm", self.node_item_name_confirm) # 确认主体
        self.workflow.add_node("node_search_embedding", self.node_search_embedding) # 向量搜索
        self.workflow.add_node("node_search_embedding_hyde", self.node_search_embedding_hyde) #假设性答案向量搜索
        self.workflow.add_node("node_web_search_mcp", self.node_web_search_mcp) # 联网搜索
        self.workflow.add_node("node_rrf", self.node_rrf) # 排序
        self.workflow.add_node("node_rerank", self.node_rerank) # 重排
        self.workflow.add_node("node_answer_output", self.node_answer_output) # 生成



    def _route_after_item_name_confirm(self, state: QueryGraphState) -> str|list[str]:
        if state.get("answer"):
            return "node_answer_output"
        else:
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
        """编译工作流（支持手动编译）"""
        if not self._compiled_graph:
            self._compiled_graph = self.workflow.compile()
        return self._compiled_graph



    def run(self, initial_state: QueryGraphState, stream: bool = False) -> QueryGraphState:
        """
        统一执行入口，支持切换invoke/stream

        Args:
            initial_state - 初始state
            stream - 流式输出开关

        Returns:
            state - 更新后state

        """
        if not self._compiled_graph:
            self.compile()

        self._compiled_graph.get_graph().print_ascii()

        if stream:
            return self._compiled_graph.stream(initial_state)
        else:
            return self._compiled_graph.invoke(initial_state)





if __name__ == "__main__":

    # 定义初始状态
    init_state = { "original_query": "如何使用万用表测量电压？"}

    graph = KBQueryWorkflow()
    # final_state = workflow.run(init_state)
    # logger.info(final_state)
    # 流式输出
    for chunk in graph.run(init_state):
        logger.warning(chunk)
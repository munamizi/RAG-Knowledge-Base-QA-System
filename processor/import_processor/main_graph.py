from langgraph.constants import END, START
from langgraph.graph import StateGraph
from processor.import_processor.nodes.node_bge_embedding import NodeBGEEmbedding
from processor.import_processor.nodes.node_document_split import NodeDocumentSplit
from processor.import_processor.nodes.node_entry import NodeEntry
from processor.import_processor.nodes.node_import_milvus import NodeImportMilvus
from processor.import_processor.nodes.node_item_name_recognition import NodeItemNameRecognition
from processor.import_processor.nodes.node_md_img import NodeMDImg
from processor.import_processor.nodes.node_pdf_to_md import NodePDFToMD
from processor.import_processor.state import ImportGraphState, get_default_state


class KBImportWorkflow:
    """创建图"""

    def __init__(self, config=None):
        # 编译好的图
        self._compiled_graph = None

    @property
    def graph(self):
        """编译好的图（如果未编译则编译）"""
        if self._compiled_graph is None:
            self._compiled_graph = self.build_graph()
        return self._compiled_graph

    def build_graph(self):
        """
        创建并编译图，返回编译好的图
        """
        # 1.创建图实例
        graph = StateGraph(ImportGraphState)

        # 2.添加节点
        graph.add_node("node_entry", NodeEntry())
        graph.add_node("node_pdf_to_md", NodePDFToMD())
        graph.add_node("node_md_img", NodeMDImg())
        graph.add_node("node_document_split", NodeDocumentSplit())
        graph.add_node("node_item_name_recognition", NodeItemNameRecognition())
        graph.add_node("node_bge_embedding", NodeBGEEmbedding())
        graph.add_node("node_import_milvus", NodeImportMilvus())

        # 3.添加边
        # 条件边
        graph.add_conditional_edges(
            "node_entry",
            self.route_after_entry,
            {
                "node_md_img": "node_md_img",
                "node_pdf_to_md": "node_pdf_to_md",
                END: END
            }
        )
        # 普通边
        graph.add_edge(START,"node_entry")
        graph.add_edge("node_pdf_to_md", "node_md_img")
        graph.add_edge("node_md_img", "node_document_split")
        graph.add_edge("node_document_split", "node_item_name_recognition")
        graph.add_edge("node_item_name_recognition", "node_bge_embedding")
        graph.add_edge("node_bge_embedding", "node_import_milvus")
        graph.add_edge("node_import_milvus", END)

        # 4.返回编译好的图
        return graph.compile()



    @staticmethod
    def route_after_entry(state: ImportGraphState) -> str:
        """
        条件函数
        """
        if state.get("is_pdf_read_enabled"):
            return "node_pdf_to_md"
        elif state.get("is_md_read_enabled"):
            return "node_md_img"
        else:
            return END

    def run(self, state: ImportGraphState, stream: bool = False):
        """
        封装图的执行，可控制invoke和stream模式：
        """
        if stream:
            return self.graph.stream(state,stream_mode="updates")
        else:
            return self.graph.invoke(state)

if __name__ == "__main__":

    init_state = get_default_state()
    graph = KBImportWorkflow()
    graph.build_graph()
    graph.run(init_state,stream=False)
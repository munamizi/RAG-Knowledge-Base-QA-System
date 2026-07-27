"""
状态结构模块
"""

from typing import TypedDict, List
import copy


class ImportGraphState(TypedDict):
    """
    导入流程图状态

    包含整个导入流程中传递的所有数据。
    使用 total=False 表示所有字段都是可选的。
    """

    # ==================== 任务标识 ====================
    task_id: str                    # 上传资料的 任务ID

    # ==================== 控制标志 ====================
    is_md_read_enabled: bool        # 被node_entry节点应用：文件为MD格式标记
    is_pdf_read_enabled: bool       # 被node_entry节点应用：文件为PDF格式标记

    # ==================== 路径信息 ====================
    import_file_path: str           # 被node_entry节点应用：获取文件路径
    file_dir: str                   # 被node_pdf_to_md节点应用：导入(出)文件目录
    pdf_path: str                   # 被node_pdf_to_md节点应用：PDF文件路径
    md_path: str                    # 被node_pdf_to_md节点应用：MD文件路径

    # ==================== 文件信息 ====================
    file_title: str                 # 被node_entry节点应用：文件名stem
    item_name: str                  # 识别出的商品/产品名称

    # ==================== 处理中间数据 ====================
    md_content: str                 # 被node_pdf_to_md节点应用：MD文档内容
    chunks: List                    # 文档切片列表



# 图默认状态
GRAPH_DEFAULT_STATE: ImportGraphState = {

    "task_id": "",

    "is_pdf_read_enabled": True,

    "is_md_read_enabled": False,

    "file_dir": r"C:\Users\Zong6\Desktop\MinerU转换结果",

    "import_file_path": r"C:\Users\Zong6\Desktop\04-设备手册汇总\doc\hak180使用说明书.pdf",

    "pdf_path": "",

    "md_path": "",

    "file_title": "",

    "md_content": "",

    "chunks": [],

    "item_name": "",

}
def get_default_state() -> ImportGraphState:
    """
    获取默认状态副本(深拷贝容器防止污染)
    """
    return copy.deepcopy(GRAPH_DEFAULT_STATE)
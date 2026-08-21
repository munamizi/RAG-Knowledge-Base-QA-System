from typing import TypedDict, List
import copy


class ImportGraphState(TypedDict):
    """导入图的状态"""

    task_id: str                    # 上传资料的任务id
    import_file_path: str           # 上传资料的路径

    is_md_read_enabled: bool        # md格式标记
    md_path: str                    # md文件路径
    is_pdf_read_enabled: bool       # pdf格式标记
    pdf_path: str                   # pdf文件路径

    file_title: str                 # 上传资料的文件名
    file_dir: str                   # 工作目录(下载MinerU压缩包的目录、解压的根目录)

    item_name: str                  # 识别出的商品名称

    md_content: str                 # md文档内容
    chunks: List                    # 文档切片列表



# 图默认状态
GRAPH_DEFAULT_STATE: ImportGraphState = {

    "task_id": "",
    "import_file_path": r"C:\Users\Zong6\Desktop\04-设备手册汇总\doc\hak180使用说明书.pdf",

    "is_pdf_read_enabled": True,
    "pdf_path": "",
    "is_md_read_enabled": False,
    "md_path": "",

    "file_title": "",
    "file_dir": r"C:\Users\Zong6\Desktop\MinerU转换结果",

    "item_name": "",

    "md_content": "",
    "chunks": [],

}


def get_default_state() -> ImportGraphState:
    """获取图默认状态"""
    return copy.deepcopy(GRAPH_DEFAULT_STATE)
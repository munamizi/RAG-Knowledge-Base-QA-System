"""
`导入图`和`查询图`任务状态管理模块：
    1.对于导入图，正常通过函数更新进下面的数据结构中
    2.对于查询图（流式），把状态内容push进session_id对应队列中，供后续SSE后端发送给前端

    数据结构：
        ①任务整体状态_tasks_status（pending、processing、completed / failed）
        例：
        {
            "task_01": "processing",
            "task_02": "completed","
        }

        ②正在执行节点_tasks_running_list
        例：
        {
            "task_01": ["node_pdf_to_md"],
            ......
        }

        ③执行完成节点_tasks_done_list
        例：
        {
            "task_01": [ "node_entry", "node_pdf_to_md"],
            ......
        }

        ④任务最终结果_tasks_result
        例：
        {
            "session_01": {"answer": "这是最终回答"},
            ......
        }


    配套方法：
        add_running_task - 添加 "正在执行节点"
        add_done_task - 添加 "执行完成节点"
        set_task_result - 存储 "任务结果"字段

        get_task_status - 获取 "任务状态"
        get_running_task_list - 获取 "正在执行节点"
        get_done_task_list - 获取 "执行完成节点"
        get_task_result - 获取 "任务结果"字段

        update_task_status - 更新 "任务状态"

        clear_task - 清除任务
"""
from typing import Any, Dict, List
from .sse_util import push_to_session


# —————————————————————————————————————————————————  数据结构 ———————————————————————————————————————————————————
_tasks_status: Dict[str, str] = {}                  # 任务状态
_tasks_running_list: Dict[str, List[str]] = {}      # 正在执行节点
_tasks_done_list: Dict[str, List[str]] = {}         # 执行完成节点
_tasks_result: Dict[str, Dict[str, str]] = {}       # 任务结果

# 任务状态
TASK_STATUS_PENDING = "pending"
TASK_STATUS_PROCESSING = "processing"
TASK_STATUS_COMPLETED = "completed"
TASK_STATUS_FAILED = "failed"

# 节点中文名映射表
_NODE_NAME_TO_CN: Dict[str, str] = {

    # —— Import 流程 ——
    "upload_file": "开始上传文件",
    "node_entry": "检查文件",
    "node_pdf_to_md": "PDF转Markdown",
    "node_md_img": "Markdown图片处理",
    "node_item_name_recognition": "主体名称识别",
    "node_document_split": "文档切分",
    "node_bge_embedding": "向量生成",
    "node_import_kg": "导入知识图谱",
    "node_import_milvus": "导入向量库",
    "__end__": "处理完成",
    "END": "处理完成",

    # —— Query 流程 ——
    "node_item_name_confirm": "确认问题产品",
    "node_answer_output": "生成答案",
    "node_rerank": "重排序",
    "node_rrf": "倒排融合",
    "node_web_search_mcp": "网络搜索",
    "node_search_embedding": "切片搜索",
    "node_search_embedding_hyde": "HyDE搜索",
    "node_multi_search": "多路搜索",
    "node_query_kg": "查询知识图谱",
    "node_join": "多路搜索合并",
}



# —————————————————————————————————————————————————  配套函数 ———————————————————————————————————————————————————
def _ensure_task(task_id: str) -> None:
    """确保 task_id 对应的任务数据结构存在。"""
    if task_id not in _tasks_running_list:
        _tasks_running_list[task_id] = []
    if task_id not in _tasks_done_list:
        _tasks_done_list[task_id] = []
    if task_id not in _tasks_result:
        _tasks_result[task_id] = {}

def _to_cn(node_name: str) -> str:
    """ 将节点名转换为中文 """
    return _NODE_NAME_TO_CN.get(node_name, node_name)


def add_running_task(task_id: str, node_name: str, is_stream: bool = False) -> None:
    """ 添加 "正在执行节点" """
    _ensure_task(task_id)
    running = _tasks_running_list[task_id]

    if node_name not in running:
        running.append(node_name)

    if is_stream:
        task_push_queue(task_id)

def add_done_task(task_id: str, node_name: str, is_stream: bool = False) -> None:
    """ 添加 "执行完成节点"，并删除对应 "正在执行节点" """
    _ensure_task(task_id)

    running = _tasks_running_list[task_id]
    _tasks_running_list[task_id] = [n for n in running if n != node_name]

    done = _tasks_done_list[task_id]
    if node_name not in done:
        done.append(node_name)

    if is_stream:
        task_push_queue(task_id)

def set_task_result(task_id: str, key: str, value: Any) -> None:
    """存储 task_id 任务结果字段。"""
    _ensure_task(task_id)
    _tasks_result[task_id][key] = value

def get_task_result(task_id: str, key: str, default: Any = "") -> Any:
    """获取 task_id 任务结果对应字段（如 answer / error）。"""
    _ensure_task(task_id)
    return _tasks_result.get(task_id, {}).get(key, default)

def get_task_status(task_id: str) -> str:
    """获取 task_id 对应 任务状态"""
    return _tasks_status.get(task_id, "")

def get_running_task_list(task_id: str) -> List[str]:
    """获取 task_id 对应 "正在执行节点" 列表。"""
    _ensure_task(task_id)
    running = _tasks_running_list.get(task_id, [])
    return [_to_cn(n) for n in running]

def get_done_task_list(task_id: str) -> List[str]:
    """获取 task_id 对应 "执行完成节点" 列表。"""
    _ensure_task(task_id)
    done = _tasks_done_list.get(task_id, [])
    return [_to_cn(n) for n in done]


def update_task_status(task_id: str, status_name: str, push_queue: bool = False) -> None:
    """更新 task_id 对应任务状态。"""
    _tasks_status[task_id] = status_name

    if status_name in {
        TASK_STATUS_COMPLETED,
        TASK_STATUS_FAILED,
    }:
        _tasks_running_list[task_id] = []

    if push_queue:
        task_push_queue(task_id)


def task_push_queue(task_id: str):
    push_to_session(
        task_id,
        "progress",
        {
            "status": get_task_status(task_id),
            "done_list": get_done_task_list(task_id),
            "running_list": get_running_task_list(task_id),
        }
    )


def clear_task(task_id: str):
    """ 清除task_id 对应 所有任务状态 """
    _tasks_running_list.pop(task_id, None)
    _tasks_done_list.pop(task_id, None)
    _tasks_status.pop(task_id, None)
    _tasks_result.pop(task_id, None)

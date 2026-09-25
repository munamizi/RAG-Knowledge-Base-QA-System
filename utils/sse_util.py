"""
SSE工具模块：
    1.定义SSE的事件类型
    2.维护session_id队列字典
    3.定义SSE后端生成器
"""
import json
import queue
import asyncio
from typing import Dict, Any, Optional, AsyncGenerator
from fastapi import Request

# ——————————————————————————————————————  定义的SSE事件类型  ———————————————————————————————————————
class SSEEvent:
    READY = "ready"         # 建立连接
    PROGRESS = "progress"   # 查询图节点执行进度
    DELTA = "delta"         # LLM 流式输出
    FINAL = "final"         # 最终完整答案和图片
    ERROR = "error"         # 错误
    CLOSE = "__close__"     # 生成器关闭信号


# ——————————————————————————————————————  session_id队列字典和配套函数  ———————————————————————————————————————
# session_id队列字典
_session_stream_dict: Dict[str, "queue.Queue"] = {}


def create_sse_queue(session_id: str) -> "queue.Queue":
    """创建 指定session_id的队列"""
    q = queue.Queue()
    _session_stream_dict[session_id] = q
    return q

def get_sse_queue(session_id: str) -> Optional["queue.Queue"]:
    """获取 指定session_id队列"""
    return _session_stream_dict.get(session_id)

def push_to_session(session_id: str, event: str, data: Dict[str, Any]):
    """存放数据到 指定session_id队列 """
    stream_queue = get_sse_queue(session_id)
    if stream_queue:
        stream_queue.put({"event": event, "data": data})

def remove_sse_queue(session_id: str):
    """删除 指定session_id的队列"""
    _session_stream_dict.pop(session_id, None)


def _sse_pack(event: str, data: Dict[str, Any]) -> str:
    """打包 SSE 消息格式"""
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


# ——————————————————————————————————————  定义SSE后端生成器  ———————————————————————————————————————
async def sse_generator(session_id: str, request: Request) -> AsyncGenerator[str, None]:

    stream_queue = get_sse_queue(session_id)
    if stream_queue is None:
        return

    try:
        # 发送初始连接建立信号
        yield _sse_pack("ready", {})

        while True:
            # 若客户端断开，尽快退出
            if await request.is_disconnected():
                break

            try:
                # 使用 run_in_executor 避免阻塞 async 事件循环
                msg = await asyncio.get_running_loop().run_in_executor(None, stream_queue.get, True, 1.0)
            except queue.Empty:
                continue

            event = msg.get("event")
            data = msg.get("data")

            # 关闭事件
            if event == "__close__":
                break

            # 发送
            yield _sse_pack(event, data)

    except (asyncio.CancelledError, ConnectionResetError, BrokenPipeError):
        return
    finally:
        remove_sse_queue(session_id)
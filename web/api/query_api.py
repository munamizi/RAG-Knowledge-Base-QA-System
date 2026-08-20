"""
    "知识问答"的全部接口：
    1. 创建分组路由实例
    2. 用户提问（执行查询图）
    3. 历史会话管理
"""
import uuid
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks, APIRouter
from pydantic import BaseModel, Field
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import FileResponse, StreamingResponse

from processor.query_processor.main_graph import KBQueryWorkflow
from utils.mongo_history_utils import clear_history, get_recent_messages
from utils.sse_util import create_sse_queue, push_to_session, SSEEvent, sse_generator
from utils.task_util import update_task_status, TASK_STATUS_PROCESSING, get_task_result, TASK_STATUS_COMPLETED, \
    TASK_STATUS_FAILED, clear_task

# 创建分组路由实例
query_router = APIRouter(
    tags=["知识问答"]
)



# 用户提问（执行查询图）API
class QueryRequest(BaseModel):
    query: str = Field(..., description="查询内容")
    session_id: str = Field(None, description="会话ID")
    is_stream: bool = Field(False, description="是否流式返回")
@query_router.post("/query")
async def query(background_tasks: BackgroundTasks, request: QueryRequest):
    """
    步骤：
        ①解析参数
        ②更新任务状态
        ③判断是否为流式输出(流失则提交后台任务，非流式则同步阻塞执行)，执行查询图，返回响应

    Args:
        background_tasks - 后台任务
        request - 本次请求

    Returns:
        响应
    """
    # 获取用户提问、session_id、流式输出标志
    user_query = request.query
    session_id = request.session_id if request.session_id else str(uuid.uuid4())
    is_stream = request.is_stream
    # 清理该会话上轮查询的任务状态
    clear_task(session_id)

    if is_stream:
        create_sse_queue(session_id)

    # 更新任务状态
    update_task_status(session_id, TASK_STATUS_PROCESSING, is_stream)

    print("开始处理流程... 是否流式:", is_stream, f"其他参数:{user_query}, session_id:{session_id}")

    if is_stream:

        # 如果是流式，则提交后台任务执行查询图；返回响应
        background_tasks.add_task(run_query_graph, session_id, user_query, is_stream)
        return {
            "message":"结果正在处理中...",
            "session_id":session_id
        }

    else:
        # 如果非流式，直接执行查询图；返回响应
        run_query_graph(session_id, user_query, is_stream)
        answer = get_task_result(session_id, "answer", "")
        return {
            "message":"处理完成！",
            "session_id":session_id,
            "answer":answer,
            "done_list":[]
        }




# 具体后台任务（流式输出）
def run_query_graph(session_id: str, user_query: str, is_stream: bool = True):
    print(f"开始流程图处理...{session_id} {user_query} {is_stream}")

    init_state = {
        "original_query": user_query,
        "session_id": session_id,
        "is_stream": is_stream
    }

    try:
        workflow = KBQueryWorkflow()
        workflow.run(init_state, stream=False)
        update_task_status(session_id, TASK_STATUS_COMPLETED, is_stream)
    except Exception as e:
        print(f"流程执行异常: {e}")
        update_task_status(session_id, TASK_STATUS_FAILED, is_stream)
        if is_stream:
            push_to_session(session_id, SSEEvent.ERROR, {"error": str(e)})




# SSE后端API
@query_router.get("/stream/{session_id}")
async def stream(session_id: str, request: Request):
    """SSE得到后端查询图节点执行情况"""
    return StreamingResponse(
        sse_generator(session_id, request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )




# 清空历史对话API
@query_router.delete("/history/{session_id}")
async def clear_chat_history(session_id: str):
    """清空当前历史对话"""
    count = clear_history(session_id)
    return {"message": "历史会话已清空", "deleted_count": count}


# 获取历史对话API
@query_router.get("/history/{session_id}")
async def history(session_id: str, limit: int = 50):
    """加载当前历史对话记录"""
    try:
        records = get_recent_messages(session_id, limit=limit)
        items = []
        for r in records:
            items.append({
                "_id": str(r.get("_id")) if r.get("_id") is not None else "",
                "session_id": r.get("session_id", ""),
                "role": r.get("role", ""),
                "text": r.get("text", ""),
                "rewritten_query": r.get("rewritten_query", ""),
                "item_names": r.get("item_names", []),
                "image_urls": r.get("image_urls") or [],
                "ts": r.get("ts")
            })
        return {"session_id": session_id, "items": items}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"history error: {e}")



# 检查服务器连接API
@query_router.get("/health")
async def health():
    """检查服务是否正常"""
    return {"ok": True}


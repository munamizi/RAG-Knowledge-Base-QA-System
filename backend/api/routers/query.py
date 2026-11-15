"""
"知识问答"的全部API：
    1.加载历史对话A
    2.query - 用户提问接口
    3.SSE实时获取后台节点执行情况API（节点进度、答案增量）
    4.清空历史对话API
    5.检查后端服务是否连接API
"""
import uuid
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks, APIRouter
from pydantic import BaseModel, Field
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import FileResponse, StreamingResponse
from starlette.concurrency import run_in_threadpool

from processor.query_processor.main_graph import KBQueryWorkflow
from utils.mongo_history_utils import clear_history, get_recent_messages
from utils.sse_util import create_sse_queue, push_to_session, SSEEvent, sse_generator
from utils.task_util import update_task_status, TASK_STATUS_PROCESSING, get_task_result, TASK_STATUS_COMPLETED, \
    TASK_STATUS_FAILED, clear_task, get_task_status


query_router = APIRouter(
    tags=["知识问答"]
)



class QueryRequest(BaseModel):
    query: str = Field(..., description="查询内容")
    session_id: str = Field(None, description="会话id")
    is_stream: bool = Field(False, description="流式开关")
@query_router.post("/query", summary="用户提问接口")
async def query(background_tasks: BackgroundTasks, request: QueryRequest):
    """
    用户提问接口
    步骤：
        ①解析参数
        ②更新任务状态
        ③判断是否为流式输出执行查询图(是则提交后台任务，否则同步阻塞执行)，返回响应

    Args:
        background_tasks - 后台任务
        request - 请求体
    """
    # 解析请求体参数（用户问题、session_id、流式开关）
    user_query = request.query
    session_id = request.session_id if request.session_id else str(uuid.uuid4())
    is_stream = request.is_stream
    clear_task(session_id)

    if is_stream:
        create_sse_queue(session_id)

    # 更新任务状态
    update_task_status(session_id, TASK_STATUS_PROCESSING, is_stream)

    print("开始处理流程... 是否流式:", is_stream, f"其他参数:{user_query}, session_id:{session_id}")

    # 流式:提交后台任务执行查询图，通过SSE后端返回响应
    if is_stream:
        background_tasks.add_task(run_query_graph, session_id, user_query, is_stream)
        return {
            "message":"结果正在处理中...",
            "session_id":session_id
        }

    # 非流式:直接执行查询图，返回响应
    else:
        await run_in_threadpool(run_query_graph, session_id, user_query, is_stream)
        if get_task_status(session_id) == TASK_STATUS_FAILED:
            raise HTTPException(status_code=502, detail="问答生成失败，请稍后重试。")
        answer = get_task_result(session_id, "answer", "")
        return {
            "message":"处理完成！",
            "session_id":session_id,
            "answer":answer,
            "citations": get_task_result(session_id, "citations", []),
            "citation_warnings": get_task_result(session_id, "citation_warnings", []),
            "done_list":[]
        }

# 流式对应具体后台任务
def run_query_graph(session_id: str, user_query: str, is_stream: bool = True):
    """初始化状态，执行检索图"""
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
    finally:
        if is_stream:
            push_to_session(session_id, SSEEvent.CLOSE, {})




@query_router.get("/stream/{session_id}", summary='SSE llm流式输出内容')
async def stream(session_id: str, request: Request):

    return StreamingResponse(
        sse_generator(session_id, request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )



@query_router.delete("/history/{session_id}", summary="清空历史对话")
async def clear_chat_history(session_id: str):
    count = clear_history(session_id)
    return {"message": "历史会话已清空", "deleted_count": count}



@query_router.get("/history/{session_id}", summary="加载历史对话记录")
async def history(session_id: str, limit: int = 50):
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
                "citations": r.get("citations") or [],
                "citation_warnings": r.get("citation_warnings") or [],
                "ts": r.get("ts")
            })
        return {"session_id": session_id, "items": items}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"history error: {e}")



@query_router.get("/health",summary="检查服务器连接")
async def health():
    return {"ok": True}


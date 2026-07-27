"""
    "导入资料"的全部接口：
    1. 创建分组路由实例
    2. 文件上传（执行检索图）
    3. 任务状态查询

"""
from datetime import datetime
import os
import shutil
import uuid
from typing import List, Dict, Any

import uvicorn
from fastapi import FastAPI, HTTPException, Path, BackgroundTasks, UploadFile, File, APIRouter
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import FileResponse

from processor.config_plus.minio_config import minio_config
from processor.import_processor.main_graph import KBImportWorkflow
from utils.logger import logger
from utils.minio_util import get_minio_client
from utils.task_util import update_task_status, add_done_task, add_running_task, get_task_status, get_done_task_list, \
    get_running_task_list

# 1. 创建分组路由实例
import_router = APIRouter(
    tags=["资料导入"]
)



# 2. 文件上传（执行检索图）
@import_router.post("/upload", summary="文件上传接口", description="支持多文件批量上传，触发后台任务：执行导入图")
async def upload_files(background_tasks: BackgroundTasks, files: List[UploadFile] = File(...)):
    """
    内容：
        1.创建该上传文件的唯一任务id
        2.创建对应的文件目录
        3.将文件保存在本地，上传到MinIO
        4.实时更新任务状态
        5.执行导入图

    Args:
        background_tasks
        files

    Returns:
        包含上传结果和所有任务ID的JSON响应

    """
    # 1. 构建本地存储根目录（按日期分层）
    data_based_root_dir = os.getenv("DATA_BASED_ROOT_DIR")
    data_dir = os.path.join(data_based_root_dir, datetime.now().strftime("%Y%m%d"))
    # 初始化任务ID列表（一个文件对应一个TaskID）
    task_ids = []

    # 2. 遍历处理每个上传的文件（各自生成对应TaskID）
    for file in files:
        # 生成全局唯一任务id，并添加到任务ID列表
        task_id = str(uuid.uuid4())
        task_ids.append(task_id)
        logger.info(f"[{task_id}]:开始处理上传文件，文件名：{file.filename}，文件类型：{file.content_type}")

        # 3. 标记「文件上传」阶段为「运行中」，前端轮询可查
        add_running_task(task_id, "upload_file")

        # 4. 构建该任务的本地独立目录，避免多文件重名冲突
        file_dir = os.path.join(data_dir, task_id)
        os.makedirs(file_dir, exist_ok=True)  # 创建对应目录，存在则不处理
        # 构建上传文件的本地保存路径
        import_file_path = os.path.join(file_dir, file.filename)

        # 5. 将上传的文件保存到本地临时目录（供后续MinIO上传/文件解析）
        with open(import_file_path, "wb") as file_buffer:
            shutil.copyfileobj(file.file, file_buffer)
        logger.info(f"[{task_id}] 文件已保存至本地，路径：{import_file_path}")

        # 6. 将本地文件上传至MinIO，做持久化保存
        # 构建要存储的对象名：pdf_files/YYYYMMDD/文件名（按日期分层，和本地一致）
        minio_object_name = f"source_files/{datetime.now().strftime('%Y%m%d')}/{file.filename}"
        try:
            # 获取MinIO客户端实例
            minio_client = get_minio_client()

            # 获取要存储的桶名
            minio_bucket_name = minio_config.bucket_name

            # 本地文件上传至MinIO（同名文件会自动覆盖，保证文件最新）
            minio_client.fput_object(
                bucket_name=minio_bucket_name,
                object_name=minio_object_name,
                file_path=import_file_path,
                content_type=file.content_type  # 传递文件原始MIME类型
            )
            logger.info(f"[{task_id}] 文件已成功上传至MinIO，桶名：{minio_bucket_name}，对象名：{minio_object_name}")
        except Exception as e:
            # MinIO上传失败，记录警告日志，继续处理本地文件
            logger.warning(f"[{task_id}] 文件上传MinIO失败，将继续执行本地处理流程，异常信息：{str(e)}", exc_info=True)

        # 7. 标记「文件上传」阶段为「已完成」，前端轮询可查
        add_done_task(task_id, "upload_file")

        # 8. 将LangGraph全流程处理加入FastAPI后台任务（异步执行，不阻塞当前接口响应）
        background_tasks.add_task(run_graph_task, task_id, file_dir, import_file_path)
        logger.info(f"[{task_id}] 已将LangGraph全流程加入后台任务，任务已启动")

    logger.info(f"多文件上传处理完毕，共处理{len(files)}个文件，生成TaskID列表：{task_ids}")
    return {
        "code": 200,
        "message": f" 文件上传成功, total: {len(files)}",
        "task_ids": task_ids
    }




# 具体后台任务
def run_graph_task(task_id: str, file_dir: str, import_file_path: str):
    """
    步骤：
        ①初始化状态
        ②流式执行导入图   每完成一个节点，将节点名加入done_list，供前端轮询查看
        ③更新任务状态  pending → processing → completed或failed

    Args:
        task_id: 本次资料导入为任务id
        file_dir: 资料输出目录（含临时文件/解析结果）
        import_file_path: 刚上传的原始文件路径

    """
    try:
        # 更新任务状态为："processing"
        update_task_status(task_id, "processing")

        # ①初始化LangGraph状态
        init_state = {
            "task_id": task_id,
            "file_dir": file_dir,
            "import_file_path": import_file_path,
        }

        # ②流式执行导入图（stream模式：实时获取每个节点的执行结果）
        workflow = KBImportWorkflow()
        for event in  workflow.run(init_state, stream=True):
            for node_name, node_result in event.items():
                # 更新已完成节点，供前端轮询/status/{task_id}获取
                add_done_task(task_id, node_name)

        # ③全流程执行完成，更新任务状态为："completed"
        update_task_status(task_id, "completed")

    except Exception as e:
        # 捕获异常，更新任务状态为："failed"，并记录日志
        update_task_status(task_id, "failed")
        logger.info(f"[{task_id}] 导入图执行失败，异常信息：{str(e)}", exc_info=True)




# 5. 任务状态查询
@import_router.get("/status/{task_id}", summary="任务状态查询", description="根据TaskID查询单个文件的处理进度态")
async def get_task_progress(task_id: str):
    """
    任务状态查询接口
    前端轮询此接口（如每秒1次），获取任务的实时处理进度
    返回数据均来自内存中的任务管理字典（task_utils.py），高性能无IO

    Args:
        task_id
    Returns:
        包含任务全局状态、已完成节点、运行中节点的JSON响应
    """
    # 构造任务状态返回体
    task_status_info: Dict[str, Any] = {
        "code": 200,
        "task_id": task_id,                             # 任务id
        "status": get_task_status(task_id),             # 任务状态：pending/processing/completed/failed
        "done_list": get_done_task_list(task_id),       # 已完成的节点列表
        "running_list": get_running_task_list(task_id)  # 正在运行的节点列表
    }
    # 记录日志
    logger.info(
        f"[{task_id}] 任务状态查询，当前状态：{task_status_info['status']}，已完成节点：{task_status_info['done_list']}")
    return task_status_info

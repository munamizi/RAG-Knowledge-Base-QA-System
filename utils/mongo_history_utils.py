import logging
import os
from datetime import datetime
from typing import List, Dict, Any

from bson import ObjectId
from dotenv import load_dotenv
from pymongo import MongoClient, ASCENDING, DESCENDING

load_dotenv(override=True)

class HistoryMongoTool:
    """
    封装MongoDB的数据库、集合、索引创建，为上层提供统一的数据库操作入口
    """
    def __init__(self):
        """
        类初始化方法：完成MongoDB的连接、数据库,集合获取、索引创建
        初始化失败会抛出异常并记录错误日志，确保程序感知连接问题
        """
        try:

            self.mongo_url = os.getenv("MONGO_URL","")
            self.db_name = os.getenv("MONGO_DB_NAME","")

            # 创建MongoDB客户端实例
            self.client = MongoClient(self.mongo_url)
            # 获取数据库
            self.db = self.client[self.db_name]
            # 获取对话记录集合
            self.chat_message = self.db["chat_message"]

            # 为chat_message集合创建复合索引以提升查询性能
            # 索引规则：session_id升序 + ts降序，适配"按会话查最新记录"的业务需求
            # create_index自带幂等性：索引已存在时不会重复创建，无需额外判断
            self.chat_message.create_index([("session_id", 1), ("ts", -1)])


            logging.info(f"Successfully connected to MongoDB: {self.db_name}")
        except Exception as e:
            logging.error(f"Failed to connect to MongoDB: {e}")
            # 重新抛出异常，让调用方感知初始化失败，避免使用未初始化的实例
            raise

# 定义全局变量存储HistoryMongoTool的实例
_history_mongo_tool = HistoryMongoTool()

def get_history_mongo_tool() -> HistoryMongoTool:
    """
    获取HistoryMongoTool的单例实例
    Returns:
        HistoryMongoTool的单例实例
    """
    global _history_mongo_tool

    if _history_mongo_tool is None:
        _history_mongo_tool = HistoryMongoTool()
    return _history_mongo_tool




def clear_history(session_id: str) -> int:
    """
    清空指定会话的所有历史对话记录
    Args:
        session_id: 会话唯一标识，用于筛选要删除的记录
    Returns:
        实际删除的文档数量，删除失败返回0
    """

    mongo_tool = get_history_mongo_tool()
    try:
        result = mongo_tool.chat_message.delete_many({"session_id": session_id})
        logging.info(f"Deleted {result.deleted_count} messages for session {session_id}")
        return result.deleted_count
    except Exception as e:
        logging.error(f"Error clearing history for session {session_id}: {e}")
        return 0




def save_chat_message(
        session_id: str,
        role: str,
        text: str,
        rewritten_query: str = "",
        item_names: List[str] = None,
        image_urls: List[str] = None,
        message_id: str = ""
) -> str:
    """
    写入/更新单条会话记录到chat_message集合

    Args:
        session_id: 会话唯一标识
        role: 消息角色，固定值：user（用户）/assistant（助手）
        text: 对话核心内容
        rewritten_query: 重写后的查询语句（可选，用于检索增强等场景，默认空字符串）
        item_names: 关联的商品名称列表（可选，支持多商品，默认None）
        image_urls: 关联的图片URL列表（可选，默认None）
        message_id: 主键ID（可选，有值则更新，无值则新增）

    Returns:
        插入/更新的记录唯一标识（新增返回ObjectId字符串，更新返回传入的message_id）
    """
    document = {
        "session_id": session_id,                   # 会话id
        "role": role,                               # 消息角色
        "text": text,                               # 消息内容
        "rewritten_query": rewritten_query or "",   # 问题优化后的改写，空值处理为空字符串
        "item_names": item_names,                   # 关联商品名称列表
        "image_urls": image_urls,                   # 关联图片URL列表
        "ts": datetime.now().timestamp()            # 时间戳（秒级），记录消息的创建时间，用于排序和查询
    }

    mongo_tool = get_history_mongo_tool()

    if message_id:
        # 有message_id执行更新
        result = mongo_tool.chat_message.update_one(
            {"_id": ObjectId(message_id)},      #  更新条件：id匹配（需将str转ObjectId类型）
            {"$set": document}                #  更新操作：$set表示只更新指定字段，保留其他字段
        )
        return message_id       # 更新操作返回传入的message_id
    else:
        # 无message_id执行新增
        result = mongo_tool.chat_message.insert_one(document)
        return str(result.inserted_id)  # 新增操作返回插入的ObjectId并转为字符串




def update_message_item_names(ids: List[str], item_names: List[str]) -> int:
    """
    批量更新历史会话记录的关联商品名称
    Args:
        ids: 要更新BSON文档的对应id列表
        item_names: 要设置的新商品名称列表

    """

    mongo_tool = get_history_mongo_tool()
    try:
        object_ids = [ObjectId(i) for i in ids]
        # 执行批量更新操作
        result = mongo_tool.chat_message.update_many(
            {
                "_id": {"$in": object_ids}
            },
            {"$set": {"item_names": item_names}}
        )

        logging.info(f"Updated {result.modified_count} records to item_names: {item_names}")

    except Exception as e:
        logging.error(f"Error updating history item_names: {e}")




def get_recent_messages(session_id: str, limit: int = 10) -> List[Dict[str, Any]]:
    """
    查询指定会话的最近N条对话记录，返回原始字典格式
    结果按时间正序排列，可直接喂给LLM作为上下文
    Args:
        session_id: 会话唯一标识，用于筛选指定会话的记录
        limit: 限制条数，默认返回最近10条
    Returns:
        对话记录列表（字典格式），查询失败返回空列表
    """

    mongo_tool = get_history_mongo_tool()
    try:
        query = {"session_id": session_id}
        # 执行查询：按时间戳升序排序，限制返回条数
        cursor = mongo_tool.chat_message.find(query).sort("ts", DESCENDING).limit(limit)
        messages = list(cursor)
        messages.reverse()
        return messages

    except Exception as e:
        logging.error(f"Error getting recent messages: {e}")
        return []
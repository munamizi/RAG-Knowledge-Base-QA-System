"""
    获取milvus客户端（全局单例模式）
"""
from pymilvus import MilvusClient

from processor.config_plus.milvus_config import milvus_config

mvilvus_client = None

def get_milvus_client():

    global mvilvus_client
    if mvilvus_client is not None:
        return mvilvus_client
    mvilvus_client = MilvusClient(uri=milvus_config.milvus_url)

    return mvilvus_client
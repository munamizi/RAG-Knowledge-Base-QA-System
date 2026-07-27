# knowledge/processor/import_process/config.py

"""
项目配置模块

    知识点：
        ① “@dataclass”装饰器：避免在一个init初始化方法中写大量的实例变量赋值；
        对于类中可变类型的实例变量，为了防止该类的实例对该可变类型实例变量共享，@dataclass要求用field=(default_factory=无参构造函数)来隔离
        ② 如果要在类中写一个类方法创建给类的实例，用cls()创建，这样在继承体系下鲁棒性更高


"""

from dataclasses import dataclass, field
from typing import Set, Optional
import os
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()


@dataclass
class ImportConfig:
    """导入流程配置"""

    # ==================== 文档处理配置 ====================
    max_content_length: int = 2000      # 切片最大长度
    img_content_length: int = 200       # 图片上下文最大长度
    min_content_length: int = 500       # 合并短内容的最小长度
    overlap_sentences: int = 1          # 句子级切分时重叠句数
    item_name_chunk_k: int = 3          # 主体识别节点使用的切片数量
    item_name_chunk_size: int = 2500    # 主体识别节点使用的切片长度上限

    # 支持的图片扩展名
    image_extensions: Set[str] = field(
        default_factory=lambda: {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
    )

    # ==================== LLM 配置 ====================
    llm_api_key: str = field(
        default_factory=lambda: os.getenv("API_KEY", "")
    )
    llm_base_url: str = field(
        default_factory=lambda: os.getenv("BASE_URL", "")
    )
    default_model: str = field(
        default_factory=lambda: os.getenv("LLM_DEFAULT_MODEL", "")
    )
    item_model: str = field(
        default_factory=lambda: os.getenv("ITEM_MODEL", "")
    )
    item_model_template:str = field(
        default_factory=lambda: os.getenv("ITEM_MODEL_TEMPERATURE", "")
    )

    # ==================== Milvus 配置 ====================
    milvus_url: str = field(
        default_factory=lambda: os.getenv("MILVUS_URL", "")
    )
    chunks_collection: str = field(
        default_factory=lambda: os.getenv("CHUNKS_COLLECTION", "")
    )
    item_name_collection: str = field(
        default_factory=lambda: os.getenv("ITEM_NAME_COLLECTION", "")
    )

    # ==================== MinIO 配置 ====================
    minio_endpoint: str = field(
        default_factory=lambda: os.getenv("MINIO_ENDPOINT", "")
    )
    minio_access_key: str = field(
        default_factory=lambda: os.getenv("MINIO_ACCESS_KEY", "")
    )
    minio_secret_key: str = field(
        default_factory=lambda: os.getenv("MINIO_SECRET_KEY", "")
    )
    minio_bucket: str = field(
        default_factory=lambda: os.getenv("MINIO_BUCKET_NAME", "")
    )
    minio_secure: bool = False

    # ==================== 向量配置 ====================
    embedding_dim: int = field(
        default_factory=lambda: int(os.getenv("EMBEDDING_DIM", "1024"))
    )
    embedding_batch_size: int = 8

    # ==================== 速率限制 ====================
    requests_per_minute: int = 15       # 图片总结 API 速率限制

# ————————————————————————————————————————————————————————————————————————————————————————————————————————————————
    @classmethod
    def from_env(cls) -> "ImportConfig":
        """类方法获取实例"""
        return cls()

    def get_minio_base_url(self) -> str:
        """获取 MinIO 基础 URL"""
        protocol = "https://" if self.minio_secure else "http://"
        return protocol + f"{self.minio_endpoint}"


# ==================== 全局单例 ====================
_config: Optional[ImportConfig] = None

def get_config() -> ImportConfig:
    """工厂模式获取配置单例"""
    global _config
    if _config is None:
        _config = ImportConfig.from_env()
    return _config

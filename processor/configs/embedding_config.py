import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass
class EmbeddingConfig:
    bge_m3_path: str
    bge_m3: str
    bge_device: str
    bge_fp16: bool
    batch_size: int

embedding_config = EmbeddingConfig(
    bge_m3_path=os.getenv("BGE_M3_PATH"),
    bge_m3=os.getenv("BGE_M3"),
    bge_device=os.getenv("BGE_DEVICE"),
    bge_fp16=os.getenv("BGE_FP16") in ("1", "True", "true", 1), # 保证结果永远是布尔值而非字符串
    batch_size=int(os.getenv("IMPORT_EMBEDDING_BATCH_SIZE", "48")),
)

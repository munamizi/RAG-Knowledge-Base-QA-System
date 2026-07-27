import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv(override=True)

@dataclass
class TruncateConfig:
    RERANK_MAX_TOPK: int     # 动态 TopK 硬上限
    RERANK_MIN_TOPK: int     # 最小 TopK：至少保留前 N 条
    RERANK_GAP_ABS: float    # 断崖阈值（绝对差值）
    RERANK_GAP_RATIO: float  # 断崖阈值（相对差值）


truncate_config = TruncateConfig(
    RERANK_MAX_TOPK = int(os.getenv("RERANK_MAX_TOPK")),
    RERANK_MIN_TOPK = int(os.getenv("RERANK_MIN_TOPK")),
    RERANK_GAP_ABS = float(os.getenv("RERANK_GAP_ABS")),
    RERANK_GAP_RATIO = float(os.getenv("RERANK_GAP_RATIO")),
)
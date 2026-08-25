from typing import Self

from pydantic import Field, model_validator

from processor.configs.base import AppBaseSettings


class TruncateConfig(AppBaseSettings):
    rerank_max_topk: int = Field(gt=0, validation_alias="RERANK_MAX_TOPK")
    rerank_min_topk: int = Field(gt=0, validation_alias="RERANK_MIN_TOPK")
    rerank_gap_abs: float = Field(ge=0, validation_alias="RERANK_GAP_ABS")
    rerank_gap_ratio: float = Field(ge=0, validation_alias="RERANK_GAP_RATIO")

    @model_validator(mode="after")
    def validate_topk_range(self) -> Self:
        if self.rerank_min_topk > self.rerank_max_topk:
            raise ValueError("RERANK_MIN_TOPK 不能大于 RERANK_MAX_TOPK")
        return self


truncate_config = TruncateConfig()

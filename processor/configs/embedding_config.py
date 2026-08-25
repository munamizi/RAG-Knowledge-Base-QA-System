from pydantic import Field

from processor.configs.base import AppBaseSettings


class EmbeddingConfig(AppBaseSettings):
    bge_m3_path: str = Field(min_length=1, validation_alias="BGE_M3_PATH")
    bge_m3: str = Field(min_length=1, validation_alias="BGE_M3")
    bge_device: str = Field(default="cpu", min_length=1, validation_alias="BGE_DEVICE")
    bge_fp16: bool = Field(default=False, validation_alias="BGE_FP16")
    batch_size: int = Field(
        default=48,
        gt=0,
        validation_alias="IMPORT_EMBEDDING_BATCH_SIZE",
    )


embedding_config = EmbeddingConfig()

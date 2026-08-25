from pydantic import Field

from processor.configs.base import AppBaseSettings


class ItemRecognitionConfig(AppBaseSettings):
    chunk_count: int = Field(default=3, gt=0, validation_alias="IMPORT_ITEM_NAME_CHUNK_K")
    max_context_length: int = Field(
        default=2500,
        gt=0,
        validation_alias="IMPORT_ITEM_NAME_CHUNK_SIZE",
    )
    model: str = Field(min_length=1, validation_alias="ITEM_MODEL")
    temperature: float = Field(
        default=0.1,
        ge=0,
        le=2,
        validation_alias="ITEM_MODEL_TEMPERATURE",
    )
    api_key: str = Field(min_length=1, repr=False, validation_alias="API_KEY")
    base_url: str = Field(min_length=1, validation_alias="BASE_URL")


item_recognition_config = ItemRecognitionConfig()

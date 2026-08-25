from pydantic import Field

from processor.configs.base import AppBaseSettings


class MilvusConfig(AppBaseSettings):
    milvus_url: str = Field(min_length=1, validation_alias="MILVUS_URL")
    chunks_collection: str = Field(min_length=1, validation_alias="CHUNKS_COLLECTION")
    item_name_collection: str = Field(min_length=1, validation_alias="ITEM_NAME_COLLECTION")


milvus_config = MilvusConfig()

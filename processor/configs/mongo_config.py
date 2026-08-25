from pydantic import Field

from processor.configs.base import AppBaseSettings


class MongoConfig(AppBaseSettings):
    mongo_url: str = Field(min_length=1, validation_alias="MONGO_URL")
    db_name: str = Field(min_length=1, validation_alias="MONGO_DB_NAME")


mongo_config = MongoConfig()

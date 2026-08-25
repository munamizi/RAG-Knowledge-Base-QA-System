from pydantic import Field

from processor.configs.base import AppBaseSettings


class MinIOConfig(AppBaseSettings):
    endpoint: str = Field(min_length=1, validation_alias="MINIO_ENDPOINT")
    access_key: str = Field(min_length=1, repr=False, validation_alias="MINIO_ACCESS_KEY")
    secret_key: str = Field(min_length=1, repr=False, validation_alias="MINIO_SECRET_KEY")
    bucket_name: str = Field(min_length=1, validation_alias="MINIO_BUCKET_NAME")
    img_dir: str = Field(min_length=1, validation_alias="MINIO_IMG_DIR")


minio_config = MinIOConfig()

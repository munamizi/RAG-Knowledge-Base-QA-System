from pydantic import Field

from processor.configs.base import AppBaseSettings


class DataStorageConfig(AppBaseSettings):
    data_based_root_dir: str = Field(min_length=1, validation_alias="DATA_BASED_ROOT_DIR")


data_storage_config = DataStorageConfig()

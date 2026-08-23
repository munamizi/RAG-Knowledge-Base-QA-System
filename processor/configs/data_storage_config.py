import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


@dataclass
class DataStorageConfig:
    data_based_root_dir: str


data_storage_config = DataStorageConfig(
    data_based_root_dir=os.getenv("DATA_BASED_ROOT_DIR"),
)

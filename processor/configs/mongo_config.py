import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv(override=True)


@dataclass
class MongoConfig:
    mongo_url: str
    db_name: str


mongo_config = MongoConfig(
    mongo_url=os.getenv("MONGO_URL", ""),
    db_name=os.getenv("MONGO_DB_NAME", ""),
)

import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


@dataclass
class ItemRecognitionConfig:
    chunk_count: int
    max_context_length: int
    model: str
    temperature: float
    api_key: str
    base_url: str


item_recognition_config = ItemRecognitionConfig(
    chunk_count=int(os.getenv("IMPORT_ITEM_NAME_CHUNK_K", "3")),
    max_context_length=int(os.getenv("IMPORT_ITEM_NAME_CHUNK_SIZE", "2500")),
    model=os.getenv("ITEM_MODEL", ""),
    temperature=float(os.getenv("ITEM_MODEL_TEMPERATURE", "0.1")),
    api_key=os.getenv("API_KEY", ""),
    base_url=os.getenv("BASE_URL", ""),
)

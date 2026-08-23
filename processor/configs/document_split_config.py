import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


@dataclass
class DocumentSplitConfig:
    max_content_length: int
    min_content_length: int


document_split_config = DocumentSplitConfig(
    max_content_length=int(os.getenv("IMPORT_MAX_CONTENT_LENGTH", "2500")),
    min_content_length=int(os.getenv("IMPORT_MIN_CONTENT_LENGTH", "500")),
)

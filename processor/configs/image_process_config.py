import os
from dataclasses import dataclass
from dotenv import load_dotenv


load_dotenv()


@dataclass
class ImageProcessConfig:
    context_length: int
    request_limit: int
    request_window_seconds: int
    supported_extensions: set[str]


image_process_config = ImageProcessConfig(
    context_length=int(os.getenv("IMPORT_IMG_CONTENT_LENGTH", "100")),
    request_limit=int(os.getenv("IMPORT_IMAGE_REQUEST_LIMIT", "30")),
    request_window_seconds=int(
        os.getenv("IMPORT_IMAGE_REQUEST_WINDOW_SECONDS", "30")
    ),
    supported_extensions={
        extension.strip().lower()
        for extension in os.getenv(
            "IMPORT_IMAGE_EXTENSIONS",
            ".jpg,.jpeg,.png,.gif,.bmp,.webp",
        ).split(",")
        if extension.strip()
    },
)

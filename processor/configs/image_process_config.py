from typing import Annotated, Any

from pydantic import Field, field_validator
from pydantic_settings import NoDecode

from processor.configs.base import AppBaseSettings


DEFAULT_IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"})


class ImageProcessConfig(AppBaseSettings):
    context_length: int = Field(
        default=200,
        gt=0,
        validation_alias="IMPORT_IMG_CONTENT_LENGTH",
    )
    request_limit: int = Field(
        default=30,
        gt=0,
        validation_alias="IMPORT_IMAGE_REQUEST_LIMIT",
    )
    request_window_seconds: int = Field(
        default=30,
        gt=0,
        validation_alias="IMPORT_IMAGE_REQUEST_WINDOW_SECONDS",
    )
    supported_extensions: Annotated[frozenset[str], NoDecode] = Field(
        default=DEFAULT_IMAGE_EXTENSIONS,
        validation_alias="IMPORT_IMAGE_EXTENSIONS",
    )

    @field_validator("supported_extensions", mode="before")
    @classmethod
    def normalize_extensions(cls, value: Any) -> frozenset[str]:
        extensions = value.split(",") if isinstance(value, str) else value
        if not isinstance(extensions, (list, tuple, set, frozenset)):
            raise ValueError("IMPORT_IMAGE_EXTENSIONS 必须是逗号分隔的扩展名列表")

        normalized = frozenset(
            f".{str(extension).strip().lower().lstrip('.')}"
            for extension in extensions
            if str(extension).strip()
        )
        if not normalized:
            raise ValueError("IMPORT_IMAGE_EXTENSIONS 不能为空")
        return normalized


image_process_config = ImageProcessConfig()

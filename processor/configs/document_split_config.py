from typing import Self

from pydantic import Field, model_validator

from processor.configs.base import AppBaseSettings


class DocumentSplitConfig(AppBaseSettings):
    max_content_length: int = Field(
        default=2500,
        gt=0,
        validation_alias="IMPORT_MAX_CONTENT_LENGTH",
    )
    min_content_length: int = Field(
        default=500,
        gt=0,
        validation_alias="IMPORT_MIN_CONTENT_LENGTH",
    )

    @model_validator(mode="after")
    def validate_length_range(self) -> Self:
        if self.min_content_length > self.max_content_length:
            raise ValueError("IMPORT_MIN_CONTENT_LENGTH 不能大于 IMPORT_MAX_CONTENT_LENGTH")
        return self


document_split_config = DocumentSplitConfig()

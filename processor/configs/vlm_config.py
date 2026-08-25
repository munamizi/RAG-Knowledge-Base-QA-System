from pydantic import Field

from processor.configs.base import AppBaseSettings


class VLMConfig(AppBaseSettings):
    base_url: str = Field(min_length=1, validation_alias="VLM_BASE_URL")
    api_key: str = Field(min_length=1, repr=False, validation_alias="VLM_API_KEY")
    vl_model: str = Field(min_length=1, validation_alias="VL_MODEL")


vlm_config = VLMConfig()

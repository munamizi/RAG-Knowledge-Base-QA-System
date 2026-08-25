from pydantic import Field

from processor.configs.base import AppBaseSettings


class LLMConfig(AppBaseSettings):
    model: str = Field(min_length=1, validation_alias="LLM_DEFAULT_MODEL")
    base_url: str = Field(min_length=1, validation_alias="BASE_URL")
    api_key: str = Field(min_length=1, repr=False, validation_alias="API_KEY")
    llm_temperature: float = Field(
        default=0.1,
        ge=0,
        le=2,
        validation_alias="LLM_TEMPERATURE",
    )


llm_config = LLMConfig()

from pydantic import Field

from processor.configs.base import AppBaseSettings


class MineruConfig(AppBaseSettings):
    base_url: str = Field(min_length=1, validation_alias="MINERU_BASE_URL")
    api_token: str = Field(min_length=1, repr=False, validation_alias="MINERU_API_TOKEN")


mineru_config = MineruConfig()

from pydantic import Field

from processor.configs.base import AppBaseSettings


class RerankerConfig(AppBaseSettings):
    text_rerank_api_key: str = Field(min_length=1, repr=False, validation_alias="API_KEY")
    text_rerank_model: str = Field(min_length=1, validation_alias="TEXT_RERANK_MODEL")
    text_rerank_instruct: str = Field(default="", validation_alias="TEXT_RERANK_INSTRUCT")


reranker_config = RerankerConfig()

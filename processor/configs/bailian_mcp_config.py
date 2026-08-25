from pydantic import Field

from processor.configs.base import AppBaseSettings


class McpConfig(AppBaseSettings):
    mcp_base_url: str = Field(min_length=1, validation_alias="MCP_DASHSCOPE_BASE_URL")
    api_key: str = Field(min_length=1, repr=False, validation_alias="API_KEY")


mcp_config = McpConfig()

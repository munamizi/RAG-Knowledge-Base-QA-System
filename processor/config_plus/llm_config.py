import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv(override=True)

@dataclass
class LLMConfig:
    model:str
    base_url:str
    api_key:str
    llm_temperature:str

llm_config = LLMConfig(
    model=os.getenv("LLM_DEFAULT_MODEL"),
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("API_KEY"),
    llm_temperature=os.getenv("LLM_TEMPERATURE")
    )

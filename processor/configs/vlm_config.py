import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv(override=True)

@dataclass
class VLMConfig:
    base_url:str
    api_key:str
    vl_model:str

vlm_config = VLMConfig(
    base_url=os.getenv("VLM_BASE_URL"),
    api_key=os.getenv("VLM_API_KEY"),
    vl_model=os.getenv("VL_MODEL")
    )
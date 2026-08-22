from langchain_openai import ChatOpenAI

from processor.configs.llm_config import llm_config


_llm_client_cache = {}

def get_llm_client(model: str | None = None, json_mode: bool = False) -> ChatOpenAI:
    """
    获取llm客户端实例

    Agrs:
        model: 模型名称
        json_mode: 是否输出为JSON

    Returns:
        model

    """
    m = model or llm_config.model
    key = (m, json_mode)
    if key in _llm_client_cache:
        return _llm_client_cache[key]


    model_kwargs: dict = {}
    if json_mode:
        model_kwargs["response_format"] = {"type": "json_object"}

    client = ChatOpenAI(
        model=m,
        temperature=llm_config.llm_temperature,
        api_key=llm_config.api_key,
        base_url=llm_config.base_url,
        extra_body={"enable_thinking": False}, # 关闭思考模式
        model_kwargs=model_kwargs,      # 控制模型生成JSON字符串
    )
    _llm_client_cache[key] = client
    return client

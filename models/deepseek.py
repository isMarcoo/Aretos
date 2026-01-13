from .base.openai_base import OpenAIBaseLLM

class DeepSeekLLM(OpenAIBaseLLM):
    def __init__(self, api_key: str, model_name: str = "deepseek-chat", **kwargs):
        # DeepSeek 官方 API 地址
        base_url = kwargs.pop("base_url", "https://api.deepseek.com")
        super().__init__(api_key, model_name, base_url, **kwargs)
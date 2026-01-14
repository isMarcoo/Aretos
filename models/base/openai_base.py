from openai import OpenAI, AsyncOpenAI
from core.base_llm import BaseLLM
from core.schemas import Message, ChatResponse, Usage
from typing import List, Any


class OpenAIBaseLLM(BaseLLM):
    def __init__(self, api_key: str, model_name: str, base_url: str, **kwargs):
        super().__init__(api_key, model_name, base_url, **kwargs)
        # 初始化 OpenAI 同步和异步客户端
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        self.async_client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)

    def chat(self, messages: List[Message], **kwargs) -> ChatResponse:
        """同步调用"""
        # 合并初始化参数和单次调用参数
        params = {**self.extra_params, **kwargs}

        # 调用 OpenAI SDK
        response = self.client.chat.completions.create(
            model=self.model_name, messages=[m.model_dump() for m in messages], **params
        )

        # 映射返回结果
        return self._parse_response(response)

    async def achat(self, messages: List[Message], **kwargs) -> ChatResponse:
        """异步调用"""
        params = {**self.extra_params, **kwargs}
        response = await self.async_client.chat.completions.create(
            model=self.model_name, messages=[m.model_dump() for m in messages], **params
        )
        return self._parse_response(response)

    def _parse_response(self, response: Any) -> ChatResponse:
        """统一解析 SDK 返回对象"""
        return ChatResponse(
            content=response.choices[0].message.content or "",
            model_name=self.model_name,
            usage=Usage(**response.usage.model_dump()),
            finish_reason=response.choices[0].finish_reason,
            raw_response=response.model_dump(),
        )

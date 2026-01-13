from openai import OpenAI
from core.base_llm import BaseLLM
from core.schemas import Message, ChatResponse, Usage

class OpenAIBaseLLM(BaseLLM):
    def __init__(self, api_key: str, model_name: str, base_url: str, **kwargs):
        super().__init__(api_key, model_name, base_url, **kwargs)
        # 初始化 OpenAI 客户端
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def chat(self, messages: List[Message], **kwargs) -> ChatResponse:
        # 合并初始化参数和单次调用参数
        generate_params = {**self.extra_params, **kwargs}
        
        # 调用 OpenAI SDK
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[m.model_dump() for m in messages],
            **generate_params
        )

        # 映射返回结果
        return ChatResponse(
            content=response.choices[0].message.content or "",
            model_name=self.model_name,
            usage=Usage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens
            ),
            raw_response=response.model_dump()
        )
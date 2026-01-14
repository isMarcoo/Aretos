from abc import ABC, abstractmethod
from typing import List, Optional
from core.schemas import Message, ChatResponse


class BaseLLM(ABC):
    def __init__(
        self, api_key: str, model_name: str, base_url: Optional[str] = None, **kwargs
    ):
        self.api_key = api_key
        self.model_name = model_name
        self.base_url = base_url
        self.extra_params = kwargs

    @abstractmethod
    def chat(self, messages: List[Message], **kwargs) -> ChatResponse:
        """
        同步对话接口
        :param messages: 符合 Message 规范的消息列表
        :param kwargs: 覆盖默认配置的额外参数（如 temperature）
        """
        pass

    @abstractmethod
    async def achat(self, messages: List[Message], **kwargs) -> ChatResponse:
        """异步对话接口，适合高并发的 Agent 场景"""
        pass

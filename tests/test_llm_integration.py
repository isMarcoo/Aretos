# tests/test_llm_integration.py
import pytest
import os
from models.deepseek import DeepSeekLLM
from models.glm import GLMLLM
from models.qwen import QwenLLM
from core.schemas import Message
from dotenv import load_dotenv

load_dotenv()


@pytest.mark.skipif(not os.getenv("QWEN_API_KEY"), reason="未设置 API Key")
def test_llm_real_chat():
    api_key = os.getenv("QWEN_API_KEY")
    llm = QwenLLM(api_key=api_key)

    messages = [Message(role="user", content="回复‘收到’")]
    response = llm.chat(messages)
    print(response)

    assert response.content is not None
    assert "收到" in response.content
    assert response.usage.total_tokens > 0


@pytest.mark.asyncio
async def test_llm_real_achat():
    api_key = os.getenv("QWEN_API_KEY")
    llm = QwenLLM(api_key=api_key)

    messages = [Message(role="user", content="你好")]
    # 测试异步接口
    response = await llm.achat(messages)
    print(response)

    assert isinstance(response.content, str)
    assert response.usage.total_tokens > 0

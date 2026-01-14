# tests/test_llm_unit.py
import os
from unittest.mock import MagicMock, patch
from core.schemas import Message
from models.deepseek import DeepSeekLLM
from dotenv import load_dotenv

load_dotenv()


def test_deepseek_payload_format():
    """测试发送给 API 的数据格式是否符合预期"""
    llm = DeepSeekLLM(api_key=os.getenv("DEEPSEEK_API_KEY"))
    messages = [Message(role="user", content="你好")]

    # 模拟客户端请求方法
    with patch("openai.resources.chat.completions.Completions.create") as mock_create:
        mock_response = MagicMock()

        mock_response.usage.model_dump.return_value = {
            "prompt_tokens": 10,
            "completion_tokens": 10,
            "total_tokens": 20,
        }

        mock_message = MagicMock()
        mock_message.content = "你好！我是 DeepSeek。"

        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_choice.finish_reason = "stop"
        mock_response.choices = [mock_choice]

        mock_response.model_dump.return_value = {
            "id": "mock_id",
            "object": "chat.completion",
        }

        mock_create.return_value = mock_response

        response = llm.chat(messages)
        # 验证调用的参数是否包含我们设定的默认值
        args, kwargs = mock_create.call_args
        assert kwargs["model"] == "deepseek-chat"
        assert kwargs["messages"][0]["content"] == "你好"
        assert response.content == "你好！我是 DeepSeek。"
        assert response.usage.total_tokens == 20
        assert os.getenv("DEEPSEEK_API_KEY") == "deepseek-xxx"

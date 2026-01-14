import logging
from models.base.openai_base import OpenAIBaseLLM

# 配置日志
logger = logging.getLogger(__name__)


class DeepSeekLLM(OpenAIBaseLLM):
    """
    DeepSeek 模型实现
    支持模型: deepseek-chat (V3), deepseek-reasoner (R1)
    """

    def __init__(
        self,
        api_key: str,
        model_name: str = "deepseek-chat",
        temperature: float = 1.0,  # DeepSeek 官方建议对话设为 1.0，代码/数学设为 0
        **kwargs,
    ):
        base_url = "https://api.deepseek.com"
        # 预设 DeepSeek 特有的默认参数
        default_params = {
            "temperature": temperature,
            "max_tokens": 4096,
            "stream": False,
        }
        default_params.update(kwargs)

        super().__init__(
            api_key=api_key, model_name=model_name, base_url=base_url, **default_params
        )
        logger.info(f"DeepSeekLLM 实例已初始化，使用模型: {model_name}")

import logging
from models.base.openai_base import OpenAIBaseLLM

# 配置日志
logger = logging.getLogger(__name__)


class MoonshotLLM(OpenAIBaseLLM):
    """
    Moonshot AI (KIMI) 模型实现
    支持模型: moonshot-v1-8k, moonshot-v1-32k, moonshot-v1-128k
    """

    def __init__(
        self,
        api_key: str,
        model_name: str = "moonshot-v1-8k",
        temperature: float = 0.3,  # KIMI 官方建议范围 (0, 1)，默认 0.3
        **kwargs,
    ):
        # Moonshot 的 OpenAI 兼容接口地址
        base_url = "https://api.moonshot.cn/v1"

        # 预设 Moonshot 特有的默认参数
        default_params = {
            "temperature": temperature,
            "max_tokens": 4096,
            "stream": False,
        }

        # 允许外部 kwargs 覆盖默认参数
        default_params.update(kwargs)

        super().__init__(
            api_key=api_key, model_name=model_name, base_url=base_url, **default_params
        )
        logger.info(f"MoonshotLLM 实例已初始化，使用模型: {model_name}")

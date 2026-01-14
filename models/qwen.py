import logging
from models.base.openai_base import OpenAIBaseLLM

# 配置日志
logger = logging.getLogger(__name__)


class QwenLLM(OpenAIBaseLLM):
    """
    通义千问 (Qwen) 模型实现
    支持模型: qwen-turbo, qwen-plus, qwen-max, qwen-long 等
    """

    def __init__(
        self,
        api_key: str,
        model_name: str = "qwen-flash",
        temperature: float = 0.7,  # Qwen 默认建议 0.7
        **kwargs,
    ):
        # 阿里云通义千问的 OpenAI 兼容接口地址
        base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"

        # 预设 Qwen 特有的默认参数
        default_params = {
            "temperature": temperature,
            "top_p": 0.8,
            "max_tokens": 1500,  # 根据需要调整
            "stream": False,
        }

        # 允许外部 kwargs 覆盖默认参数
        default_params.update(kwargs)

        super().__init__(
            api_key=api_key, model_name=model_name, base_url=base_url, **default_params
        )
        logger.info(f"QwenLLM 实例已初始化，使用模型: {model_name}")

import logging
from models.base.openai_base import OpenAIBaseLLM

# 配置日志
logger = logging.getLogger(__name__)


class GLMLLM(OpenAIBaseLLM):
    """
    智谱清言 (ChatGLM/GLM-4) 模型实现
    支持模型: glm-4, glm-4-plus, glm-4-0520, glm-4-air 等
    """

    def __init__(
        self,
        api_key: str,
        model_name: str = "glm-4.7",
        temperature: float = 0.95,  # GLM 官方建议范围 (0, 1)，默认 0.95
        **kwargs,
    ):
        # 智谱 GLM 的 OpenAI 兼容接口地址
        base_url = "https://open.bigmodel.cn/api/paas/v4/"

        # 预设 GLM 特有的默认参数
        default_params = {
            "temperature": temperature,
            "top_p": 0.7,  # GLM 官方建议 top_p 设为 0.7
            "max_tokens": 4096,
            "stream": False,
        }

        # 允许外部 kwargs 覆盖默认参数
        default_params.update(kwargs)

        super().__init__(
            api_key=api_key, model_name=model_name, base_url=base_url, **default_params
        )
        logger.info(f"GLMLLM 实例已初始化，使用模型: {model_name}")

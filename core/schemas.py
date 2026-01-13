from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Dict, Any

class Message(BaseModel):
    """标准的对话消息结构"""
    role: Literal["system", "user", "assistant"]
    content: str

class Usage(BaseModel):
    """Token 使用情况"""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

class ChatResponse(BaseModel):
    """统一的模型输出结果"""
    content: str
    model_name: str
    usage: Optional[Usage]
    raw_response: Optional[Dict[str, Any]] = None  # 保留原始响应以备查
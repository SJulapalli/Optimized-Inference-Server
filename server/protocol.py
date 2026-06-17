from pydantic import BaseModel, Field
from typing import Literal, Optional
import time
import uuid

def _ts() -> int:
    return int(time.time())

def _id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"

class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    temperature: float = 1.0
    top_p: float = -1.0
    top_k: int = -1
    max_tokens: int = 256
    stream: bool = False
    repetition_penalty: float = 1.0


class CompletionRequest(BaseModel):
    model: str
    prompt: str
    temperature: float = 1.0
    top_p: float = -1.0
    top_k: int = -1
    max_tokens: int = 256
    stream: bool = False
    repetition_penalty: float = 1.0


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=_ts)
    owned_by: str = "user"

class ModelList(BaseModel):
    object: str = "list"
    data: list[ModelCard]

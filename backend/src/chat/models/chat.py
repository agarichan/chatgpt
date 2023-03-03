from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class ChatRole(str, Enum):
    SYSTEM = "system"
    ASSISTANT = "assistant"
    USER = "user"


class ChatMessage(BaseModel):
    role: ChatRole
    content: str


class ChatRequest(BaseModel):
    model: Literal["gpt-3.5-turbo", "gpt-3.5-turbo-0301"]
    messages: list[ChatMessage]
    temperature: float | None = Field(
        default=1,
        ge=0,
        le=2,
        description=(
            "サンプリング温度は0～2の間で指定します。"
            "0.8のような高い値は出力をよりランダムにし、0.2のような低い値は出力をより集中的に決定的にします。"
            "一般的には、この変更かtop_pの変更をお勧めしますが、両方はお勧めしません。"
        ),
    )
    top_p: float | None = Field(
        default=1,
        ge=0,
        le=1,
        description=(
            "温度によるサンプリングに代わるものとして、核サンプリングと呼ばれる、確率質量がtop_pのトークンの結果を考慮するモデルがある。"
            "つまり、0.1は上位10%の確率質量からなるトークンだけを考慮することを意味します。"
        ),
    )
    n: int | None = Field(default=1, ge=0, description="各入力メッセージに対して、チャット補完の選択肢をいくつ生成するか。")
    max_tokens: int | None = Field(
        default=None, description=("生成される答えに許可されるトークンの最大数です。" "デフォルトでは、モデルが返すことのできるトークン数は（4096 - prompt tokens）となります。")
    )


class ChatUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatResponseChoice(BaseModel):
    message: ChatMessage
    finish_reason: str | None
    index: int


class ChatResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    usage: ChatUsage | None
    choices: list[ChatResponseChoice]


class ChatDelta(BaseModel):
    role: ChatRole | None
    content: str | None


class ChatStreamChoice(BaseModel):
    delta: ChatDelta
    finish_reason: str | None
    index: int


class ChatStream(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: list[ChatStreamChoice]

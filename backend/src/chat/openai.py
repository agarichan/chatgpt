import json
import os
from typing import AsyncGenerator

from aiohttp import ClientSession

from .models.chat import ChatRequest, ChatResponse, ChatStream


class AsyncOpenAI:
    def __init__(self, api_key: str | None = None) -> None:
        api_key = api_key or os.environ["OPENAI_API_KEY"]
        headers = {"Authorization": f"Bearer {api_key}"}
        self.session = ClientSession(base_url="https://api.openai.com", headers=headers)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()

    async def close(self):
        await self.session.close()

    async def chat(self, request: ChatRequest) -> ChatResponse:
        async with self.session.post("/v1/chat/completions", json=request.dict()) as res:
            return ChatResponse(**await res.json())

    async def stream_chat(self, request: ChatRequest) -> AsyncGenerator[ChatStream, None]:
        dic = request.dict() | {"stream": True}
        async with self.session.post("/v1/chat/completions", json=dic) as res:
            async for line in res.content:
                line = line.decode("utf8")
                if line == "\n" or line == "\r\n" or line == "\r":
                    continue
                line = line.lstrip("data: ").rstrip("\n")
                if line == "[DONE]":
                    break
                data = json.loads(line)
                yield ChatStream(**data)

from typing import AsyncGenerator

from .models.chat import ChatMessage, ChatRequest, ChatRole
from .openai import AsyncOpenAI


class Chat:
    MODEL = "gpt-3.5-turbo-0301"

    def __init__(self, client: AsyncOpenAI, initialize_messages: list[ChatMessage] = []):
        self.messages = initialize_messages
        self.client = client

    async def send(self, content: str) -> str:
        msg = ChatMessage(role=ChatRole.USER, content=content)
        request = ChatRequest(model=self.MODEL, messages=self.messages + [msg])
        res = await self.client.chat(request)
        self.messages += [msg]
        return res.choices[0].message.content

    async def send_generator(self, content: str) -> AsyncGenerator[str, None]:
        msg = ChatMessage(role=ChatRole.USER, content=content)
        request = ChatRequest(model=self.MODEL, messages=self.messages + [msg])
        res = ""
        async for data in self.client.stream_chat(request):
            res = ""
            c = data.choices[0].delta.content or ""
            res += c
            yield c
        self.messages += [msg]

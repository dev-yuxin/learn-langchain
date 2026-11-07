import asyncio
import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv(override=True)

llm = init_chat_model(
    api_key=os.getenv("bailian_api_key"),
    base_url=os.getenv("bailian_base_url"),
    model="openai:deepseek-v4.1-flash",
    extra_body={"thinking": {"type": "disabled"}},
)


def test_stream():
    iterator = llm.stream("什么是langchain")
    for ai_message_chunk in iterator:
        print(ai_message_chunk.content, end="", flush=True)


async def test_astream():
    aiterator = llm.astream("什么是langgraph")
    async for ai_message_chunk in aiterator:
        print(ai_message_chunk.content, end="", flush=True)


if __name__ == '__main__':
    test_stream()
    asyncio.run(test_astream())

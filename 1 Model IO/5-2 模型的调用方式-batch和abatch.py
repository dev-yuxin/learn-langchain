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


def test_batch():
    result = llm.batch(inputs=["写个笑话", "写首诗"])
    for ai_message in result:
        print(ai_message)


async def test_abatch():
    result = await llm.abatch(inputs=["写个笑话", "写首诗"])
    for ai_message in result:
        print(ai_message)


if __name__ == '__main__':
    test_batch()
    asyncio.run(test_abatch())

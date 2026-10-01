import asyncio

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv(override=True)

model = init_chat_model(model="deepseek:deepseek-flash")


async def test_abatch():
    ai_messages = await model.abatch(["你是谁", "介绍一下llm"])
    return [ai_message.text for ai_message in ai_messages]


if __name__ == "__main__":
    answers = asyncio.run(test_abatch())
    print(answers)

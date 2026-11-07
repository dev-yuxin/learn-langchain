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


# 模型调用方式（同步/异步）
# 直接调用：invoke(), ainvoke()
# 批量调用：batch(), abatch()
# 流式调用：stream(), astream()
def test_invoke():
    ai_message1 = llm.invoke("写个笑话")
    print(ai_message1)
    ai_message2 = llm.invoke("写首诗")
    print(ai_message2)


async def test_ainvoke(input):
    ai_message = await llm.ainvoke(input)
    return ai_message


async def main_task():
    task1 = asyncio.create_task(test_ainvoke("写个笑话"))
    task2 = asyncio.create_task(test_ainvoke("写首诗"))
    result = await asyncio.gather(task1, task2)
    for ai_message in result:
        print(ai_message)


if __name__ == '__main__':
    test_invoke()
    asyncio.run(main_task())

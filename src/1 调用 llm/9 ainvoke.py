import asyncio

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv(override=True)

model = init_chat_model(model="deepseek:deepseek-flash")


# ainvoke
async def async_write_novel(output_file: str):
    with open(output_file, "w", encoding="utf-8") as f:  # noqa: ASYNC230
        ai_message = await model.ainvoke("帮我写一篇500字的小说")
        f.write(ai_message.text)
    print(f"已写入{output_file}")


async def async_write_novels(*output_files: str):
    async_tasks = []
    for output_file in output_files:
        # 这里不能加 await，异步函数不执行，返回一个协程对象
        async_task = async_write_novel(output_file)
        async_tasks.append(async_task)
    await asyncio.gather(*async_tasks)


if __name__ == "__main__":
    # 两个文件是并行非流式写的
    asyncio.run(async_write_novels("./novel1.txt", "./novel2.txt"))

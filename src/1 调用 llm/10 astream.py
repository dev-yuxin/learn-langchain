import asyncio

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv(override=True)

model = init_chat_model(model="deepseek:deepseek-flash")


# astream
async def async_write_novel(output_file: str):
    with open(output_file, "w", encoding="utf-8") as f:  # noqa: ASYNC230
        astream = model.astream("帮我写一篇500字的小说")
        async for ai_message_chunk in astream:
            f.write(ai_message_chunk.content)
            f.flush()
    print(f"已写入{output_file}")


async def async_write_novels(*output_files: str):
    async_tasks = []
    for output_file in output_files:
        async_task = async_write_novel(output_file)
        async_tasks.append(async_task)
    await asyncio.gather(*async_tasks)


if __name__ == "__main__":
    # 两个文件是并行流式写的
    asyncio.run(async_write_novels("./novel1.txt", "./novel2.txt"))

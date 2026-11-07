import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, TextContentBlock, ImageContentBlock

load_dotenv(override=True)

llm = init_chat_model(
    api_key=os.getenv("bailian_api_key"),
    base_url=os.getenv("bailian_base_url"),
    model="openai:qwen3.8-flash",
)

# langchain1.0后新增content_blocks来兼容多模态内容
human_message = HumanMessage(content_blocks=[
    TextContentBlock(type="text", text="图片里是什么"),
    ImageContentBlock(
        type="image",
        # 或者用 base64 字段
        url="https://fuss10.elemecdn.com/e/5d/4a731a90594a4af544c0c25941171jpeg.jpeg",
        mime_type="image/jpeg"
    )
])
ai_message = llm.invoke(input=[human_message])

# content类型是str | list[str | dict[Any, Any]]，大部分情形下是str
print(ai_message.content)
# text类型一定是str
print(ai_message.text)

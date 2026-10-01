import base64
import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, ImageContentBlock, TextContentBlock

load_dotenv(override=True)

llm = init_chat_model(
    # 模型要支持多模态
    model="qwen3.8-max",
    model_provider="openai",
    base_url="https://api.openai-proxy.org/v1",
    api_key=os.getenv("CLOSEAI_API_KEY"),
)


# 读取本地图片并转为 base64 编码传输
def read_local_image(img_path: str):
    with open(img_path, "rb") as f:
        image_data = f.read()
        return base64.b64encode(image_data).decode()


# 多模态输入（使用 context_blocks 字段）
human_message = HumanMessage(
    content_blocks=[
        TextContentBlock(type="text", text="描述这2张图片"),
        ImageContentBlock(
            type="image",
            url="https://fuss10.elemecdn.com/e/5d/4a731a90594a4af544c0c25941171jpeg.jpeg",
            mime_type="image/jpeg",
        ),
        ImageContentBlock(
            type="image",
            base64=read_local_image(os.path.join(os.getcwd(), "assets", "zidane.jpg")),
            mime_type="image/jpeg",
        ),
    ],
)

# AIMessage 的 content 的类型是 str | list[str | dict]，大部分 llm 直接返回 str ，少部分情形返回 list
# 你可以使用 text 字段保证获取到的是字符串
ai_message = llm.invoke([human_message])
print(ai_message.text)

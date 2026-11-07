import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser

load_dotenv(override=True)

llm = init_chat_model(
    api_key=os.getenv("bailian_api_key"),
    base_url=os.getenv("bailian_base_url"),
    model_provider="openai",
    model="qwen-flash",
)

ai_message = llm.invoke("你好，你是谁")

# 将 ai_message 解析位字符串
parser = StrOutputParser()
text = parser.invoke(ai_message)
print(text)

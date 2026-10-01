import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# 读取环境变量，项目根目录下的配置将覆盖系统环境变量
load_dotenv(override=True)

llm = ChatOpenAI(
    # 必填，要和平台的模型名对应上
    model="gpt-5",
    # 这里我们用了中转站而不是官方
    # 按优先级取：显示传入 > 环境变量 OPENAI_API_BASE > 环境变量 OPENAI_BASE_URL > 官方地址
    base_url="https://api.openai-proxy.org/v1",
    # 按优先级取：显示传入 > 环境变量中OPENAI_API_KEY
    api_key=os.getenv("CLOSEAI_API_KEY"),
)

ai_message = llm.invoke(input="你好，你是谁")
print(ai_message)

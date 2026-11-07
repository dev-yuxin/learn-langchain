import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv(override=True)

# 1 创建大模型对象
llm = ChatOpenAI(
    # api_key: 显示指定 > 环境变量中的 OPENAI_API_KEY
    api_key=os.getenv("bailian_api_key"),
    # base_url: 显示指定 > 环境变量中的 OPENAI_BASE_URL > 环境变量中的 OPENAI_API_BASE > OPENAI的官方url
    base_url=os.getenv("bailian_base_url"),
    model="qwen-flash"
)

# 2 调用模型
ai_message = llm.invoke("简单介绍一下langchain")
print(ai_message)

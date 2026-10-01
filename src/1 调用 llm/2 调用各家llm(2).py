import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv(override=True)

llm = ChatOpenAI(
    model="deepseek-v4.1-flash",
    base_url="https://api.openai-proxy.org/v1",
    api_key=os.getenv("CLOSEAI_API_KEY"),
)

ai_message = llm.invoke(input="你好，你是谁")
print(ai_message)

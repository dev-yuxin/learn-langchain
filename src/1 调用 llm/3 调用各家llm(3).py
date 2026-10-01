from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek

load_dotenv(override=True)

llm = ChatDeepSeek(
    model="deepseek-flash",
    # 参数 base_url 取官方地址
    # 参数 api_key 取环境变量 DEEPSEEK_API_KEY
)

ai_message = llm.invoke(input="你好，你是谁")
print(ai_message)

import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate

load_dotenv(override=True)

llm = init_chat_model(
    api_key=os.getenv("bailian_api_key"),
    base_url=os.getenv("bailian_base_url"),
    model_provider="openai",
    model="qwen-flash",
)

# PromptTemplate 提示词模板，只能以字符串作为模板
pt = PromptTemplate.from_template("""
你是一个{role}。
你的任务是{task}。
用户输入:{input}。
""")
# 得到的是一个 PromptValue 对象
pv = pt.invoke(input={
    "role": "法律助手",
    "task": "回答用户的法律咨询",
    "input": "张三酒驾犯什么罪"
})
ai_message = llm.invoke(input=pv)
print(ai_message)

# ChatPromptTemplate 提示词模板
# Sequence[any] 可以传入元组列表、字典列表、但不能是List[BaseMessage]（无法解析占位符） 作为模板
cpt = ChatPromptTemplate.from_messages([
    ("system", "你是一个{role}，你的任务是{task}。"),
    ("human", "{input}")
])
# 得到的是一个 PromptValue 对象
pv = cpt.invoke(input={
    "role": "法律助手",
    "task": "回答用户的法律咨询",
    "input": "张三酒驾犯什么罪"
})
ai_message = llm.invoke(input=pv)
print(ai_message)

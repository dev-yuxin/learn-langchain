import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv(override=True)

llm = init_chat_model(
    api_key=os.getenv("bailian_api_key"),
    base_url=os.getenv("bailian_base_url"),
    model="openai:deepseek-v4.1-flash",
    extra_body={"thinking": {"type": "disabled"}},
)

# input 可以是
# PromptValue: 提示词模板解析后的值
# str: 字符串
# Sequence[Any]: 字典列表、元组列表、消息列表

# 传入字符串
ai_message1 = llm.invoke(input="你好，你是谁")
print(ai_message1)

# 字典形式
ai_message2 = llm.invoke(input=[
    # role 可以填 system, user, human, ai, assistant, tool
    {"role": "system", "content": "你是一个有用的助手"},
    {"role": "human", "content": "1+1=?"},
    {"role": "ai", "content": "2"},
    {"role": "human", "content": "再加1呢？"},
])
print(ai_message2)

# 元组形式
ai_message2 = llm.invoke(input=[
    ("system", "你是一个有用的助手"),
    ("human", "1+1=?"),
    ("ai", "2"),
    ("human", "再加1呢？"),
])
print(ai_message2)

# List[BaseMessage] SystemMessage|HumanMessage|AIMessage|HumanMessage
ai_message3 = llm.invoke(input=[
    SystemMessage(content="你是一个有用的助手"),
    HumanMessage(content="1+1=?"),
    AIMessage(content="2"),
    HumanMessage(content="再加1呢？")
])
print(ai_message3)

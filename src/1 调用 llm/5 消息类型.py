from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv(override=True)

llm = init_chat_model(model="deepseek:deepseek-flash")

# 传入字符串
ai_message = llm.invoke("你好，你是谁")
print(ai_message)
print("=" * 100)

# 传入消息列表
ai_message = llm.invoke(
    [SystemMessage("你是一个人工智能专家"), HumanMessage("简单介绍一下 llm")]
)
print(ai_message)
print("=" * 100)

# 传入元组型消息列表
ai_message = llm.invoke(
    [("system", "你是一个人工智能专家"), ("human", "简单介绍一下 llm")]
)
print(ai_message)
print("=" * 100)


# 传入字典型消息列表
ai_message = llm.invoke(
    [
        {"role": "system", "content": "你是一个人工智能专家"},
        {"role": "human", "content": "简单介绍一下 langgraph"},
    ]
)
print(ai_message)

import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.postgres import PostgresSaver

load_dotenv(override=True)

with PostgresSaver.from_conn_string(os.getenv("POSTGRESQL_URI")) as checkpointer:
    # 首次运行需要建表，后续不会再建表
    checkpointer.setup()

    # 带短期记忆的智能体
    agent = create_agent(
        model=init_chat_model(model="deepseek:deepseek-flash"),
        checkpointer=checkpointer,
    )
    # 需要设置 thread_Id 并把 config 传入
    config = {"configurable": {"thread_id": "thread_1"}}

    state = agent.invoke(
        input={"messages": [HumanMessage("你好，我是小明")]}, config=config
    )
    print(state["messages"][-1].text)

    state = agent.invoke(
        input={"messages": [HumanMessage("你好，我是谁")]}, config=config
    )
    print(state["messages"][-1].text)

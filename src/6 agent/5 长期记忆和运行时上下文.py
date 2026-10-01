import os
from typing import TypedDict

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import ToolRuntime, tool
from langchain_core.messages import HumanMessage
from langgraph.store.postgres import PostgresStore

load_dotenv(override=True)

# 存入长期记忆
with PostgresStore.from_conn_string(os.getenv("POSTGRESQL_URI")) as store:
    # 和短期记忆一样，首次需要执行一次
    store.setup()
    # namespace 必须是字符串元组
    store.put(
        namespace=("users", "zhangsan"),
        key="preference",
        value={"hobbies": ["唱", "跳", "篮球"]},
    )
    # value 必须是可序列化的字典
    store.put(
        namespace=("users", "lisi"),
        key="preference",
        value={"hobbies": ["动漫", "游戏"]},
    )


# 上下文类型定义
class UserContext(TypedDict):
    user_id: str


@tool
# 注意，这里 runtime 是通过类型注解注入的，agent 看不见这个参数
def get_user_hobby(runtime: ToolRuntime[UserContext]):
    """获取用户爱好"""
    user_id = runtime.context.get("user_id", "")
    item = runtime.store.get(("users", user_id), "preference")
    if item:
        return item.value.get("hobbies")


with PostgresStore.from_conn_string(os.getenv("POSTGRESQL_URI")) as store:
    store.setup()

    # agent 注入长期记忆，定义用户运行时上下文
    agent = create_agent(
        model=init_chat_model(model="deepseek:deepseek-flash"),
        tools=[get_user_hobby],
        store=store,
        context_schema=UserContext,
    )

    # 传入运行时上下文，只对本次生效
    state = agent.invoke(
        input={"messages": HumanMessage("基于我的兴趣推荐一些好玩的东西")},
        context={"user_id": "zhangsan"},
    )
    print(state["messages"][-1].text)

    state = agent.invoke(
        input={"messages": HumanMessage("基于我的兴趣推荐一些好玩的东西")},
        context={"user_id": "lisi"},
    )
    print(state["messages"][-1].text)

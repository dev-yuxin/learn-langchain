from datetime import date, datetime

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field

load_dotenv(override=True)


@tool
def get_cur_time():
    """获取当前时间"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # noqa: DTZ005


@tool
def get_weather(city: str, time: date):
    """
    获取城市某日的天气

    Args:
        city: 城市
        time: 日期（天）
    """
    return f"{city} {time} 15度，小雨"


# 参数比较复杂时使用 Pydantic
class TrainArgs(BaseModel):
    depart_city: str = Field(description="出发城市")
    arrive_city: str = Field(description="结束城市")
    depart_date: date = Field(description="出发日")


# 你也可以在 tool 中使用 description 和 args_schema，优先级比文档注释更高
@tool(description="获取某日火车票余票信息", args_schema=TrainArgs)
def get_train(depart_city: str, arrive_city: str, depart_date: date):
    return f"{depart_date} {depart_city} -> {arrive_city} 只剩 G123 一辆车，9:00 出发，二等座剩余10张，"


llm = init_chat_model(model="deepseek:deepseek-flash")

# 基于 langgraph 实现
agent = create_agent(model=llm, tools=[get_cur_time, get_weather, get_train])

state = agent.invoke(
    {"messages": [HumanMessage("帮我查一下明天北京去上海的火车，顺便看看下不下雨")]}
)
for message in state["messages"]:
    message.pretty_print()

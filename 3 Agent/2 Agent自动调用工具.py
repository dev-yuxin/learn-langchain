import os
from datetime import datetime
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field

load_dotenv(override=True)


# 工具
# 大模型识别工具依靠的是函数名+参数
# 一种方法是使用 Google-Style 的注释
# 另一种方式是在装饰器中写参数
@tool
def get_curtime():
    """获取当前时间"""
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


@tool
def get_weather(city: str, time: str):
    """
    获取城市指定时间的天气

    Args:
        city: 城市
        time: 指定时间
    """
    return f"{city} {time} 下雨，20度"


class PlayRecommend(BaseModel):
    city: str = Field(description="城市")
    outer: bool = Field(description="是否室外", default=True)


@tool(description="获取城市的游玩推荐，分为室外和室内", args_schema=PlayRecommend)
def get_play_recommend(city: str, outer=True):
    if city == "北京":
        return "故宫\n长城" if outer else "鸟巢\n水立方"
    elif city == "上海":
        return "外滩" if outer else "东方明珠塔"
    else:
        return "其它城市正在入驻中"


# 工具列表
tools = [get_curtime, get_weather, get_play_recommend]

# 大模型
llm = init_chat_model(
    api_key=os.getenv("bailian_api_key"),
    base_url=os.getenv("bailian_base_url"),
    model_provider="openai",
    model="deepseek-v4.1-flash"
)

if __name__ == '__main__':
    # agent 能自动调用工具
    agent = create_agent(
        model=llm,
        tools=tools,
    )
    output_state = agent.invoke(input={
        # 基于 langgraph 必须叫 messages
        "messages": [HumanMessage("帮我查询一下北京明天的天气，然后推荐一些游玩的地方，如果天晴则户外，下雨则室内。")]
    })
    for message in output_state["messages"]:
        message.pretty_print()

import os
from datetime import datetime
from dotenv import load_dotenv
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
tools_map = {
    "get_curtime": get_curtime,
    "get_weather": get_weather,
    "get_play_recommend": get_play_recommend,
}

# 大模型绑定工具（可以识别工具，但工具调用得显示调用，只有Agent才会自动调用工具）
llm = init_chat_model(
    api_key=os.getenv("bailian_api_key"),
    base_url=os.getenv("bailian_base_url"),
    model_provider="openai",
    model="deepseek-v4.1-flash"
)
llm_with_tools = llm.bind_tools(tools)

if __name__ == '__main__':
    message_list = [
        HumanMessage("帮我查询一下北京明天的天气，然后推荐一些游玩的地方，如果天晴则户外，下雨则室内。")
    ]
    # ReAct 架构: 思考 => 执行 => 观察 => 再思考 => 再执行
    while True:
        ai_message = llm_with_tools.invoke(message_list)
        ai_message.pretty_print()
        message_list.append(ai_message)
        if not ai_message.tool_calls:
            break
        else:
            for tool_call in ai_message.tool_calls:
                # 大模型本身是不会调用工具的，这里需要我们手动调用再把调用结果以 ToolMessage 形式给到大模型
                tool_message = tools_map[tool_call["name"]].invoke(tool_call)
                tool_message.pretty_print()
                message_list.append(tool_message)

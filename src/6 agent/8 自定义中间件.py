import random
from typing import TypedDict

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallRequest, wrap_tool_call
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool

load_dotenv(override=True)


@tool
def get_weather(city: str):
    """
    获取城市今日的天气

    Args:
        city: 城市
        time: 日期（天）
    """
    if random.random() < 0.8:
        raise RuntimeError("网络异常，请稍后重试")
    return f"{city} 今天15度，小雨"


class ToolContext(TypedDict):
    max_attempt: int


@wrap_tool_call
def retry_tool_calls(request: ToolCallRequest, handler):
    """
    工具调用失败时自动重试，最多 3 次。
    - request: ToolCallRequest, 包含 request.tool_call / request.tool / request.state 等
    - handler: 下一个处理函数（通常是真正的工具执行）
    """
    max_attempt = (request.runtime.context or {}).get("max_attempt", 3)
    tool_call_id = request.tool_call.get("id")
    tool_call_name = request.tool_call.get("name")
    exceptions = []
    for attempt in range(max_attempt):
        try:
            return handler(request)
        except Exception as e:  # noqa: BLE001
            print(f"第 {attempt + 1} 次调用工具 {tool_call_name} 失败: {e}")
            exceptions.append(e)
    return ToolMessage(
        content=f"工具 {tool_call_name} 调用次数达到上限 {max_attempt} 次，全部失败，失败原因: {exceptions}",
        tool_call_id=tool_call_id,
    )


agent = create_agent(
    model=init_chat_model("deepseek:deepseek-flash"),
    tools=[get_weather],
    middleware=[retry_tool_calls],
    context_schema=ToolContext,
)

stream = agent.stream(
    input={"messages": [HumanMessage("北京今天天气如何")]},
    stream_mode=["updates"],
    context={"max_attempt": 3},
)
for item in stream:
    messages = (item[1].get("model", {}) or item[1].get("tools", {})).get(
        "messages", []
    )
    for message in messages:
        message.pretty_print()

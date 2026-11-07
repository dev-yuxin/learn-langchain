import os
from random import random

from dotenv import load_dotenv
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import before_model, after_model, AgentMiddleware, ModelRequest
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.runtime import Runtime

load_dotenv(override=True)


# 工具
@tool
def get_weather(city: str, time: str):
    """
    获取天气

    Args:
        city: 城市
        time： 时间
    """
    if random() > 0.5:
        return f"{city} {time} 下雨，20度"
    else:
        raise RuntimeError("网络波动，请稍后再试")


tools = [get_weather]

# 大模型
llm = init_chat_model(
    api_key=os.getenv("bailian_api_key"),
    base_url=os.getenv("bailian_base_url"),
    model_provider="openai",
    model="deepseek-v4.1-flash"
)


# 自定义中间件
# node_style: 类似于 langgraph 的节点函数
#   before_agent: 开启agent前调用一次，一般用于初始化环境，身份认证等
#   after_agent: agent结束后调用一次，一般用于清理工作
#   before_model: 每次调用模型前执行，一般用于检查输入是否合法，上下文处理等
#   after_model: 每次调用模型后执行，一般用于格式化输出，记录日志，统计token等
# wrap_style
#   wrap_model_call: 调用模型前，调用模型，调用模型后
#   wrap_tool_call: 调用工具前，调用工具，调用工具后

# 如果你中间件只用到一个 hook，那么写成函数式
@before_model
def check_input(state: AgentState, runtime: Runtime):
    print("上下文处理")


@after_model
def log(state: AgentState, runtime: Runtime):
    print("记录token消耗情况")


# 如果你中间件要用到多个 hook，那么继承AgentMiddleware并重写 hook
class EnvMiddleware(AgentMiddleware):
    def before_agent(self, state: AgentState, runtime: Runtime):
        print("初始化环境")

    def after_agent(self, state: AgentState, runtime: Runtime):
        print("清理环境")


class ExecuteMiddleware(AgentMiddleware):
    def wrap_model_call(self, request: ModelRequest, handler):
        print("模型调用前...")
        need_update = True
        if need_update:
            print("升级为pro模型")
        ai_message = handler(request)
        print("模型调用后")
        # 一般返回一条 ai_message
        # 也可以像langgraph 一样返回 Command
        return ai_message

    def wrap_tool_call(
            self,
            request: ToolCallRequest,
            handler,
    ):
        max_attempt = 3
        tool_message = None
        for i in range(max_attempt):
            try:
                tool_message = handler(request)
                print(f"第{i + 1}次调用{request.tool_call['name']}成功")
                break
            except Exception as e:
                print(f"第{i + 1}次调用{request.tool_call['name']}失败: {e}")
        if not tool_message:
            tool_message = ToolMessage(
                tool_call_id=request.tool_call["id"],
                content="3次调用工具失败，不再调用",
                status="error"
            )
        return tool_message


# 智能体
agent = create_agent(
    model=llm,
    tools=tools,
    # 如果有多个中间件[m1, m2, m3]，执行顺序（正序进入，反序退出）
    # before_agent: m1 => m2 => m3
    # before_model: m1 => m2 => m3
    # after_model: m3 => m2 => m1
    # after_agent: m3 => m2 => m1
    # wrap_model_call 和 wrap_tool_call:
    #   m1 进入 (before handler)
    #     m2 进入 (before handler)
    #       m3 进入 (before handler)
    #         handler(request)  -- 实际调用模型或工具
    #       m3 退出 (after handler)
    #     m2 退出 (after handler)
    #   m1 退出 (after handler)
    middleware=[
        EnvMiddleware(),
        check_input,
        log,
        ExecuteMiddleware()
    ]
)

if __name__ == '__main__':
    iterator = agent.stream(input={
        "messages": [HumanMessage("帮我查询一下北京明天的天气，然后推荐一些游玩的地方，如果天晴则户外，下雨则室内。")]
    })
    for item in iterator:
        print(item)

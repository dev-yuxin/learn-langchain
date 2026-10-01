import ast

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command, Interrupt

load_dotenv(override=True)


# 工具
@tool
def ask_city():
    """询问用户所在城市"""
    return "北京"


@tool
def get_weather(city: str):
    """
    查询指定城市天气

    Args:
        city: 城市名称
    """
    return f"{city}今天晴，20度"


@tool
def get_news():
    """
    获取最新新闻
    """
    return "黄金价格突破一千\n美伊爆发最新冲突\n皇马夺得欧冠"


@tool
def send_email(receiver: str, title: str, content: str):
    """
    向指定邮箱发送邮件内容

    Args:
        receiver: 收件人邮箱
        title: 邮件标题
        content: 邮件内容
    """
    print(f"""已经向{receiver}发送邮件""")
    return True


# 人工审批工具调用中间件
human_in_the_loop_middleware = HumanInTheLoopMiddleware(
    interrupt_on={
        # ask_city 中断，可以选择 respond
        "ask_city": {
            "allowed_decisions": ["respond"],
            "description": "请输入你所在的城市：",
        },
        # get_weather 中断，可以选择 approve/edit/reject
        "get_weather": {
            "allowed_decisions": ["approve", "reject", "edit"],
        },
        # get_news 不中断，直接调用工具
        "get_news": False,
        # send_email 中断，可以选择 approve/reject
        "send_email": {
            "allowed_decisions": ["approve", "reject"],
        },
    }
)


# 处理中断
def deal_interrupts(interrupts: list[Interrupt]):
    """
    处理中断，返回值格式（要和中断顺序一一对应）
    [
        {
            "type": "approved"
        },
        {
            "type": "reject"
            "message": "拒绝原因"  # 可选
        },
        {
            "type": "edit",
            "edited_action": {
                "name": tool_call_name,
                "args": new_args  # json 格式
            }
        },
        {
            "type": "respond",
            "message": "工具返回结果"
        }
    ]
    """
    decisions = []

    for interrupt in interrupts:
        action_requests = interrupt.value["action_requests"]
        review_configs = interrupt.value["review_configs"]

        for action_request, review_config in zip(action_requests, review_configs):
            print("=" * 100)

            tool_call_name = action_request["name"]
            tool_call_args = action_request["args"]
            allowed_decisions = review_config["allowed_decisions"]

            while True:
                decision_type = (
                    input(
                        f"工具调用中断，工具名{tool_call_name}，参数{tool_call_args}，请输入处理策略（{'/'.join(allowed_decisions)}）："
                    )
                    .strip()
                    .lower()
                )
                if decision_type in allowed_decisions:
                    break
                print("输入不合法，请重新输入")

            if decision_type == "approve":
                decisions.append({"type": "approve"})

            if decision_type == "reject":
                reject_reason = input("请输入拒绝原因（可跳过）：").strip()
                decisions.append(
                    {"type": "reject", "message": reject_reason}
                    if reject_reason
                    else {"type": "reject"}
                )

            if decision_type == "respond":
                while True:
                    message = input(action_request["description"]).strip()
                    if message:
                        decisions.append({"type": "respond", "message": message})
                        break
                    print("输入不合法，请重新输入")

            if decision_type == "edit":
                while True:
                    raw = input(f"原参数{tool_call_args}，请输入新的参数：")
                    try:
                        new_args = ast.literal_eval(raw)
                        decisions.append(
                            {
                                "type": "edit",
                                "edited_action": {
                                    "name": tool_call_name,
                                    "args": new_args,
                                },
                            }
                        )
                        break
                    except:  # noqa: E722
                        print("输入不合法，请重新输入参数")
    return decisions


agent = create_agent(
    model=init_chat_model(model="deepseek:deepseek-flash"),
    tools=[ask_city, get_weather, get_news, send_email],
    middleware=[human_in_the_loop_middleware],
    # 中断依赖于 checkpointer
    checkpointer=InMemorySaver(),
)

config = {"configurable": {"thread_id": "thread1"}}

stream = agent.stream(
    input={
        "messages": [
            HumanMessage(
                "请帮我查询今天的天气和最新新闻，然后向 xiaoming@qq.com 发送邮件。标题是今日快讯，内容是你查询到的天气和新闻。其中地理位置变化可以忽略。"
            )
        ]
    },
    config=config,
    stream_mode=["updates"],
)


while True:
    interrupts = []

    for item in stream:
        data = item[1]

        if "__interrupt__" in data:
            interrupts.extend(data["__interrupt__"])

        messages = (data.get("model", {}) or data.get("tools", {})).get("messages", [])
        for message in messages:
            message.pretty_print()

    # 没有中断就终止
    if not interrupts:
        break

    # 有中断进入 human in the loop
    decisions = deal_interrupts(interrupts)

    stream = agent.stream(
        input=Command(resume={"decisions": decisions}),
        config=config,
        stream_mode=["updates"],
    )

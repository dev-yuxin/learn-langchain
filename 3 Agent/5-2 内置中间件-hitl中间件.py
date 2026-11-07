import ast
import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command, Interrupt

load_dotenv(override=True)


@tool
def get_weather(city: str, time: str):
    """
    获取城市指定时间的天气

    Args:
        city: 城市
        time: 指定时间
    """
    return f"{city} {time} 下雨，20度"


@tool
def send_email(address: str, title, content: str):
    """
    发送邮件

    Args:
        address: 邮箱地址
        title: 标题
        content: 内容
    """
    print("发送成功")
    return True


llm = init_chat_model(
    api_key=os.getenv("bailian_api_key"),
    base_url=os.getenv("bailian_base_url"),
    model_provider="openai",
    model="qwen-flash"
)

# 人工审核中间件
hitl_middleware = HumanInTheLoopMiddleware(
    # interrupt_on 格式
    # {
    #     "tool1": True,
    #     "tool2": False,
    #     "tool3": {
    #         "allowed_decisions": ["approve", "reject", "edit", "respond"],
    #         "description": ""
    #     }
    # }
    interrupt_on={
        "get_weather": False,
        "send_email": {
            "allowed_decisions": ["approve", "reject"],
            "description": "发送邮件需要审批"
        }
    }
)

# 中断依赖于 checkpointer
agent = create_agent(
    model=llm,
    tools=[get_weather, send_email],
    middleware=[hitl_middleware],
    checkpointer=InMemorySaver()
)
config = {
    "configurable": {
        "thread_id": "123"
    }
}


# 处理中断
def deal_interrupts(interrupts: list[Interrupt]):
    resume_map = {}
    for interr in interrupts:
        # hitl_request 格式
        # {
        #     "action_requests": [
        #         {
        #             "name": 'send_email',
        #             "args": {},
        #             "description": ""
        #         }
        #     ],
        #     "review_configs": [
        #         {
        #             "action_name": "send_email",
        #             "allowed_decisions": ["approve", "reject"]
        #         }
        #     ]
        # }
        hitl_request = interr.value
        # hitl_response 格式
        # {
        #     "decisions": [
        #         {
        #             "type": "approve"
        #         },
        #         {
        #             "type": "edit",
        #             "edited_action": {
        #                 "args": {}
        #             }
        #         },
        #         {
        #             "type": "reject",
        #             "message": "拒绝原因"
        #         },
        #         {
        #             "type": "respond",
        #             "message": "直接替代工具返回结果"
        #         },
        #     ]
        # }
        hitl_response = {"decisions": []}
        for action, review_cfg in zip(hitl_request.get("action_requests", []), hitl_request.get("review_configs", [])):
            print(action.get("description", "工具需要审批"))
            allowed_decisions = review_cfg.get(
                "allowed_decisions", ["approve", "reject", "edit", "respond"]
            )
            while True:
                decision_type = input(f"请输入({'/'.join(allowed_decisions)}): ").strip().lower()
                if decision_type not in allowed_decisions:
                    print("输入错误，请重新输入")
                else:
                    break

            if decision_type == "approve":
                decision = {"type": "approve"}
                hitl_response["decisions"].append(decision)
            elif decision_type == "reject":
                message = input("请输入拒绝原因（可跳过）").strip()
                decision = {"type": "reject"}
                if message:
                    decision["message"] = message
                hitl_response["decisions"].append(decision)
            elif decision_type == "edit":
                while True:
                    new_args_str = input("请输入新的参数（字典字符串格式）")
                    try:
                        new_args = ast.literal_eval(new_args_str)
                        break
                    except Exception as e:
                        print("输入错误，请重新输入", e)
                decision = {
                    "type": "edit",
                    "edited_action": {
                        "args": new_args
                    }
                }
                hitl_response["decisions"].append(decision)
            elif decision_type == "respond":
                message = input("请输入工具结果").strip()
                decision = {
                    "type": "respond",
                    "message": message
                }
                hitl_response["decisions"].append(decision)
            else:
                decision = {
                    "type": "respond",
                    "message": "无效审批类型，跳过工具"
                }
                hitl_response["decisions"].append(decision)
        resume_map[interr.id] = hitl_response
    return resume_map


if __name__ == '__main__':
    iterator = agent.stream(
        input={
            "messages": [
                HumanMessage(
                    "请帮我查询北京明天天气，然后发送到邮箱xiaoming@123.com，标题是今日天气，内容是你查询到的结果")
            ]
        },
        config=config
    )

    while True:
        interrupts = []
        # 打印消息并收集中断
        for item in iterator:
            messages = (item.get("model") or item.get("tools") or {}).get("messages", [])
            for message in messages:
                message.pretty_print()
            interrupts += item.get("__interrupt__", [])
        if not interrupts:
            break
        else:
            resume = deal_interrupts(interrupts)
            iterator = agent.stream(
                input=Command(resume=resume),
                config=config
            )

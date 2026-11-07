import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv(override=True)

llm = init_chat_model(
    api_key=os.getenv("bailian_api_key"),
    base_url=os.getenv("bailian_base_url"),
    model_provider="openai",
    model="qwen-flash"
)

# 记忆摘要中间件，会把上下文总结成一条消息
summ_middleware = SummarizationMiddleware(
    model=llm,
    # 触发时机（或的关系，只要满足任一一个就触发）
    trigger=[
        ("messages", 5),
        ("tokens", 10000),
    ],
    # 保留最近3条消息，其余做摘要
    keep=("messages", 3),
)

agent = create_agent(
    model=llm,
    middleware=[summ_middleware],
)

if __name__ == '__main__':
    output_state = agent.invoke(input={
        "messages": [
            HumanMessage("1+2=?"),
            AIMessage("3"),
            HumanMessage("1+2+3=?"),
            AIMessage("6"),
            HumanMessage("1+2+...+100=?"),
            AIMessage("5050"),
            HumanMessage("给出求和公式"),
        ]
    })
    for message in output_state["messages"]:
        message.pretty_print()

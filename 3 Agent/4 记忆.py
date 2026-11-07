import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv(override=True)

# 大模型
llm = init_chat_model(
    api_key=os.getenv("bailian_api_key"),
    base_url=os.getenv("bailian_base_url"),
    model_provider="openai",
    model="deepseek-v4.1-flash"
)

# 请参见 langgraph
# 短期记忆：
#   当前会话中直接进入模型上下文窗口的信息，比如最近几轮对话、系统提示、工具结果。
#   短期记忆依赖于会话id
#   langchain和langgraph中短期记忆是指checkpointer，可以存内存里，也可以存数据库
# 长期记忆：
#   存在模型上下文之外、跨会话持久保留、需要时检索回来的信息，比如用户偏好、历史摘要、事实档案、知识片段。
#   长期记忆是跨会话的
#   langchain和langgraph中长期记忆是指store，可以存内存里，也可以存数据库
agent = create_agent(
    model=llm,
    # 设置短期记忆，存内存里
    checkpointer=InMemorySaver()
)

# 配置信息，固定格式
config = {
    "configurable": {
        "thread_id": "123"  # 短期记忆依赖于会话id
    }
}

if __name__ == '__main__':
    output_state = agent.invoke(input={
        "messages": [("human", "你好我是张三")]
    }, config=config)
    for message in output_state["messages"]:
        message.pretty_print()

    output_state = agent.invoke(input={
        "messages": [("human", "我是谁")]
    }, config=config)
    for message in output_state["messages"]:
        message.pretty_print()

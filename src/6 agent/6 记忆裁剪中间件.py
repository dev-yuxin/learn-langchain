from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv(override=True)

# 基于内存的短期记忆
checkpointer = InMemorySaver()

# 摘要中间件
summarization_middleware = SummarizationMiddleware(
    # 用于摘要的模型
    model=init_chat_model(model="deepseek:deepseek-flash"),
    # 触发时机，三个任一一个满足
    # ("messages", 5): 消息数达到5条触发
    # ("fraction", 0.5): token达到上下文窗口的50%
    # ("tokens", 10000): token数量达到10000
    trigger=[("tokens", 10000), ("messages", 5), ("fraction", 0.5)],
    # 保留最后2条消息，前面的内容进行摘要总结为一条用户消息，只能写一个保留条件
    keep=("messages", 2),
    # 摘要最大token数（溢出将截断取最后面的），默认400
    trim_token_to_summarize=4000,
)

agent = create_agent(
    model=init_chat_model(model="deepseek:deepseek-flash"),
    middleware=[summarization_middleware],
)

state = agent.invoke(
    {
        "messages": [
            HumanMessage("你好，你是谁"),
            AIMessage("我是一个AI助手，我能为你在编程，日常生活等领域提供帮助"),
            HumanMessage("你好，我是小明"),
            AIMessage("你好，小明，有什么需要我帮助你的吗"),
            HumanMessage("介绍一下 langchain"),
            AIMessage("LangChain 是一个用于构建大语言模型（LLM）应用的开源框架。"),
            HumanMessage("介绍一下 langgraph"),
        ]
    }
)
for message in state["messages"]:
    message.pretty_print()

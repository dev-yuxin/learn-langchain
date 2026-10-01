from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv(override=True)

llm = init_chat_model(
    model="deepseek-flash",
    model_provider="deepseek",
    # 指定在运行时可以用 configurable 覆盖的字段，是个元组
    configurable_fields=("model",) 
)

ai_message = llm.invoke(
    input="你好",
    # 运行时配置
    config={
        # 一些给 langSmith 调试、追踪、日志用的字段
        "tags": ["demo"],
        "metadata": {"user_id": "123"},
        # configurable 本次调用改变模型参数
        "configurable": {
            "model": "deepseek-v4-pro",
        },
    },
)

print(ai_message)

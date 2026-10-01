from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv(override=True)

llm = init_chat_model(
    # 你可以直接写 model="deepseek:deepseek-v4-flash" 从而省略 model_provider
    # model_provider 底层会去调用对应的 ChatXxx
    model="deepseek-flash",
    model_provider="deepseek",
    # 温度 [0, 2] 值越小越确定（写代码，数学问题），值越大越有创造性（写文案等），默认0.7
    temperature=1.8,
    # 最大输出长度（单位是token）
    max_tokens=24,
    # 超时时间（秒）
    timeout=60,
    # 失败重试次数
    max_retries=2,
    # openai 支持但 langchain 未列出的字段放到 model_kwargs 中透传
    # openai 不支持，但模型额外支持的字段，比如 deepseek 支持关闭思考模式
    extra_body={"thinking": {"type": "disabled"}},
)

ai_message = llm.invoke("请帮我写一首七言绝句")
print(ai_message)

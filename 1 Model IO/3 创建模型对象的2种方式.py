import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_deepseek import ChatDeepSeek
from langchain_openai import ChatOpenAI

load_dotenv(override=True)

# 1 使用 ChatXxx 创建大模型对象（使用OpenAI的SDK）
llm1 = ChatOpenAI(
    # api_key: 显示指定 > 环境变量中的 OPENAI_API_KEY
    api_key=os.getenv("bailian_api_key"),
    # base_url: 显示指定 > 环境变量中的OPENAI_BASE_URL > 环境变量中的OPENAI_API_BASE > OPENAI的官方url
    base_url=os.getenv("bailian_base_url"),
    # 使用的模型，得和平台的保持一致
    model="qwen-flash"
)

# 1 使用 ChatXxx 创建大模型对象（使用Deepseek的SDK）
llm2 = ChatDeepSeek(
    # api_key: 显示指定 > 环境变量中的DEEPSEEK_API_KEY
    api_key=os.getenv("deepseek_api_key"),
    # base_url: 显示指定 > 环境变量中的DEEPSEEK_BASE_URL >  环境变量中的DEEPSEEK_API_BASE > DEEPSEEK的官方url
    base_url=os.getenv("deepseek_base_url"),
    model="deepseek-flash"
)

# 2 使用 init_chat_model 创建大模型对象
llm3 = init_chat_model(
    # 平台的 api_key 和 base_url
    api_key=os.getenv("bailian_api_key"),
    base_url=os.getenv("bailian_base_url"),
    # model_provider 可以填 openai, anthropic, google_genai, deepseek 等
    # 底层会去调用 ChatOpenAI, ChatAnthropic, ChatGoogleGenerativeAI, ChatDeepSeek 等
    # 国产模型 langchain 只集成了 DeepSeek，但几乎所有国产模型都兼容 OpenAI 协议
    model_provider="openai",
    # 使用的模型，得和平台名字保持一致
    # 你也可以简写 model="openai:deepseek-v4.1-flash" 从而省略 model_provider
    model="deepseek-v4.1-flash",
    # 温度 [0, 2] 固定 - 随机
    temperature=1,
    # 单次输出最大tokens
    max_tokens=1000,
    # 单次请求最大时间
    timeout=10,
    # 最大重试次数
    max_retries=3,
    # model_kwargs: openai 支持的其它字段组成一个字典
    # extra_body: openai 不支持，但 deepseek 模型额外支持的字段组成一个字典，比如关闭思考模式
    extra_body={"thinking": {"type": "disabled"}},
    # configurable_fields 声明可更改的配置字段，在 config={"configurable": {}} 中覆盖配置
    configurable_fields=["model"]
)

ai_message1 = llm1.invoke("你好，你是谁")
print(ai_message1)

ai_message2 = llm2.invoke("你好，你是谁")
print(ai_message2)

ai_message3 = llm3.invoke(input="你好，你是谁", config={"configurable": {"model": "qwen-flash"}})
print(ai_message3)

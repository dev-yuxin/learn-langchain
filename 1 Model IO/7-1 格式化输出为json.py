import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field

load_dotenv(override=True)

llm = init_chat_model(
    api_key=os.getenv("bailian_api_key"),
    base_url=os.getenv("bailian_base_url"),
    model_provider="openai",
    model="qwen-flash",
)


class Prime(BaseModel):
    primes: list[int] = Field(description="素数列表")
    count: int = Field(description="素数的数量")


# langchain 封装的厂商能力（大部分厂商都支持工具调用）
llm_output_json = llm.with_structured_output(schema=Prime)

# 输出的 result 是一个 Prime 对象
result = llm_output_json.invoke([
    HumanMessage("请输出1-100的素数")
])
print(result)

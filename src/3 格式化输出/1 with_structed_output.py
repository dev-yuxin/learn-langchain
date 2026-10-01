import re
from typing import Literal

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field, field_validator

load_dotenv(override=True)

llm = init_chat_model(
    model="deepseek:deepseek-flash",
    # deepseek-flash 模型思考模式不支持格式化输出，关了
    extra_body={"thinking": {"type": "disabled"}},
)


# 继承 pydantic 的 BaseModel
class Person(BaseModel):
    # 大模型能看见这些参数定义，会自己去填补修正
    # 姓名，必填，字符串型，最少2字符，最多20字符
    name: str = Field(description="姓名", min_length=2, max_length=20)
    # 年龄，必填，整形，0 <=age <= 200
    age: int = Field(description="年龄", ge=0, le=200)
    # 性别，必填，"male" or "female"
    gender: Literal["male", "female"] = Field(description="性别")
    # 联系方式，非必填，符合正则校验
    phone: str | None = Field(
        description="联系方式", pattern=r"^1[3-9]\d{9}$", default=None
    )

    # 大模型看不见这些，校验不通过会抛异常
    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str):
        """姓名中不能有数字"""
        if re.search(r"\d", value):
            raise ValueError("姓名中不能包含数字")
        return value


structed_model = llm.with_structured_output(schema=Person)

try:
    # llm 会根据约束自己去填补，如果最终转换不通过还是会抛异常
    person1 = structed_model.invoke("你好，我是小明")
    print(person1)
    person2 = structed_model.invoke("我是123")
    print(person2)
except ValueError as e:
    print(e)

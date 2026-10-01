from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate

load_dotenv(override=True)

llm = init_chat_model(model="deepseek:deepseek-flash")

# 定义样例列表
examples = [
    {"course": "大模型介绍", "importance": 1},
    {"course": "dify&coze", "importance": 2},
    {"course": "langchain", "importance": 4.5},
    {"course": "langgraph", "importance": 5},
]

# 这里的 course 和 importance 要和上面 examples 要一一对应
example_prompt = PromptTemplate.from_template(
    "课程：{course}，重要指数（满分5星）：{importance}星"
)

# 带样例的单字符串形式的提示词模板
# 前缀 + "\n\n" + 遍历 examples 生成 example_prompt + "\n\n" + 后缀
few_shot_prompt_template = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="以下是大模型相关课程",
    suffix="请帮我列出重要性>={custom_importance}星课程",
    input_variables=["custom_importance"],
)

# 格式化后返回的是 StringPromptValue
str_prompt_value = few_shot_prompt_template.invoke({"custom_importance": 4})

ai_message = llm.invoke(str_prompt_value)
print(ai_message.content)

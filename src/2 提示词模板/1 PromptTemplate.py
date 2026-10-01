from langchain_core.prompts import PromptTemplate

# 创建 PromptTemplate（单字符串形式的提示词模板）
prompt_template = PromptTemplate.from_template(
    template="请解释一下{topic}，要求使用{language}和{style}风格"
)

# 固定部分参数
prompt_template = prompt_template.partial(language="中文", style="通俗易懂")

# 格式化 PromptTemplate，返回 StringPromptValue（单字符串形式的提示词值）
str_prompt_value = prompt_template.invoke({"topic": "人工智能"})
print(str_prompt_value, type(str_prompt_value))

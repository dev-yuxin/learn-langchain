from langchain_core.prompts import ChatPromptTemplate

# 创建 ChatPromptTemplate（多轮对话提示词模板）
chat_prompt_template = ChatPromptTemplate.from_messages(
    # 你可以使用元组列表、字典列表
    # 但不能使用内置的消息列表，放到 content 中的内容不解析
    [("system", "你是一个{area}领域的专家"), ("human", "{question}")]
)

# 格式化，返回 ChatPromptValue（多轮对话形式的提示词值）
chat_prompt_value = chat_prompt_template.invoke(
    {"area": "python", "question": "什么是装饰器"}
)
print(chat_prompt_value, type(chat_prompt_value))

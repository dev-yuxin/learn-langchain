from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate

chat_prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个ai助手"),
        # 在 ChatPromptTemplate 预留占位，传入的值要求是 Sequence[MessageLikeRepresentation] 类消息序列
        # 下面是简写形式，等价于 MessagesPlaceholder(variable_name="history")
        ("placeholder", "{history}"),
        ("human", "{question}"),
    ]
)

chat_prompt_value = chat_prompt_template.invoke(
    {
        "history": [
            HumanMessage("什么是 python"),
            AIMessage("python 是一门编程语言"),
        ],
        "question": "它有什么用",
    }
)
print(chat_prompt_value)

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

load_dotenv(override=True)

# 简单顺序链
# | 管道符把上一个的输出作为下一个的输入
chain = (
    ChatPromptTemplate.from_messages(
        [
            ("system", "你是一个{area}领域的专家，你要回答用户相关问题，回答要{style}"),
            ("human", "{question}"),
        ]
    )
    | init_chat_model(model="deepseek:deepseek-flash")
    | StrOutputParser()
)

answer = chain.invoke(
    {"area": "人工智能", "style": "简洁明了", "question": "介绍一下国内外有哪些主流llm"}
)
print(answer)

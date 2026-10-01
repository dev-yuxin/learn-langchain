from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

load_dotenv(override=True)


def rag(query: str):
    """模拟rag"""
    return "我们是ai初创公司，主营企业级agent服务，公司员工有10人。"


# 并行链
chain = (
    # RunablePassthrough 将原样输出
    # 字典遇到 | 将变成一个并行结构
    # 函数遇到 | 将自动包裹变成一个 runnable 对象
    {"query": RunnablePassthrough(), "context": rag}
    | PromptTemplate.from_template("问题：{query}\n\n这是上下文背景{context}")
    | init_chat_model(model="deepseek:deepseek-chat")
    | StrOutputParser()
)

answer = chain.invoke("公司有多少人")
print(answer)

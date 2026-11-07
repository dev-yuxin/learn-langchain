import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnableSequence, RunnablePassthrough

load_dotenv(override=True)

cpt = ChatPromptTemplate.from_messages([
    ("system", "你是一名翻译，你要把用户输入翻译成{target_language}"),
    ("user", "{input}")
])

llm = init_chat_model(
    api_key=os.getenv("bailian_api_key"),
    base_url=os.getenv("bailian_base_url"),
    model_provider="openai",
    model="qwen-flash",
)

llm2 = init_chat_model(
    api_key=os.getenv("bailian_api_key"),
    base_url=os.getenv("bailian_base_url"),
    model_provider="openai",
    model="deepseek-v4.1-flash",
)

str_parser = StrOutputParser()

# 不用链式写法
pv = cpt.invoke(input={"target_language": "日语", "input": "好好学习，天天向上"})
ai_message = llm.invoke(input=pv)
text = str_parser.invoke(input=ai_message)
print(text)

# 链式写法
# langchain 中有一个类 Runnable，它里面有核心方法
# invoke()/ainvoke()
# batch()/abatch()
# stream()/astream()
# 模型类，解析器类、提示词模板类等大部分组件类都继承了 Runnable 类，并且重写了其中一些方法，大部分都会重写 invoke。
# 继承了 Runnable 类的子类对象，可以使用链式写法。

# 顺序链
# | 类似于linux的管道符，把前一个的输出作为后一个的输入
# 是 RunnableSequence(*steps) 的语法糖简写
chain = cpt | llm | str_parser
text = chain.invoke(input={"target_language": "日语", "input": "好好学习，天天向上"})
print(text)

# 组合链
# 填充部分占位符
english_cpt = cpt.partial(target_language="英语")
# {} 是 RunnableParallel({}) 的语法糖简写形式，并行链
# RunnablePassthrough() 把输入原样返回
# 并行链得到的是一个字典，作为第二步的输入，键要一一对应
chain3 = ({
              # 把输入原样返回
              "original_input": RunnablePassthrough(),
              # 使用不同模型翻译
              "trans1": english_cpt | llm | str_parser,
              "trans2": english_cpt | llm2 | str_parser
          } | PromptTemplate.from_template("""我有一首中文诗：{original_input}。
         英文翻译一：{trans1}。
         英文翻译二：{trans2}。
         哪个版本翻译的更好，为什么""") | llm | str_parser)
text = chain3.invoke(input={"input": "菩提本无树，明镜亦非台。本来无一物，何处惹尘埃。"})
print(text)

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv(override=True)

llm = init_chat_model(model="deepseek:deepseek-flash")

# invoke
ai_message = llm.invoke("你是谁")
print(ai_message.text)
print("=" * 100)
ai_message = llm.invoke("介绍一下llm")
print(ai_message.text)
print("=" * 100)

# batch
ai_messages = llm.batch(["你是谁", "介绍一下llm"])
for ai_message in ai_messages:
    print(ai_message.text)
    print("=" * 100)

# stream
ai_message_chunks = llm.stream("介绍一下llm")
for ai_message_chunk in ai_message_chunks:
    print(ai_message_chunk.text, end="", flush=True)

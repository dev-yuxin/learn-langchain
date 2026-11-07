import asyncio
import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv(override=True)

# mcp客户端
client = MultiServerMCPClient({
    # 远程 streamable_http 的 mcp
    "12306-mcp": {
        "transport": "streamable_http",
        "url": "https://mcp.api-inference.modelscope.net/0e8e68d4e51d4c/mcp",
        "headers": {
            "Authorization": f"Bearer {os.getenv('modelscope_api_token')}"
        }
    },
    # 本地 stdio 的 mcp
    "stdio-weather-mcp": {
        "transport": "stdio",
        "command": "python",
        "args": ["3-1 mcp-创建stdio服务.py"],
    }
})

# 大模型
llm = init_chat_model(
    api_key=os.getenv("bailian_api_key"),
    base_url=os.getenv("bailian_base_url"),
    model_provider="openai",
    model="deepseek-v4.1-flash"
)


# 远程 mcp 服务只支持异步，必须以异步形式调用
async def main():
    tools = await client.get_tools()
    agent = create_agent(
        model=llm,
        tools=tools,
    )
    aiterator = agent.astream(input={
        "messages": [HumanMessage("你好，请帮我查询明天北京去上海的火车票。再查询一下天气。")]
    })
    async for item in aiterator:
        messages = (item.get("model", {}) or item.get("tools", {})).get("messages", [])
        for message in messages:
            message.pretty_print()


if __name__ == '__main__':
    asyncio.run(main())

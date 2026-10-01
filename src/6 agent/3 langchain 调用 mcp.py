import asyncio
import os
import sys

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.mcp import MCPAdapter
from langchain_core.messages import HumanMessage

load_dotenv(override=True)

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "2 mcp(stdio).py")

config = {
    "mcpServers": {
        # 用 python 启动本地服务
        "my-local-tools": {
            "command": sys.executable,  # 使用当前虚拟环境的 Python
            "args": [file_path],  # 要启动的脚本文件
        },
        # 高德远程服务
        "amap-maps-streamableHTTP": {
            "url": f"https://mcp.amap.com/mcp?key={os.getenv('AMAP_KEY')}"
        },
    }
}


async def main():
    async with MCPAdapter(config) as adapter:
        tools = await adapter.list_tools()
        agent = create_agent(
            model=init_chat_model(model="deepseek:deepseek-v4-flash"),
            tools=tools,
        )
        astream = agent.astream(
            input={
                "messages": [
                    HumanMessage("我明天从苏州开车去上海，大概需要多久，另外下雨吗")
                ]
            },
            stream_mode=["updates"],
        )
        async for item in astream:
            messages = (item[1].get("model", {}) or item[1].get("tools", {})).get(
                "messages", []
            )
            for message in messages:
                message.pretty_print()


if __name__ == "__main__":
    asyncio.run(main())

from datetime import datetime

from mcp.server.fastmcp import FastMCP

# 创建 MCP 实例
mcp = FastMCP()


# 为 MCP 实例添加工具
@mcp.tool()
def get_weather(city: str, time: str):
    """获取城市指定时间的天气"""
    return f"{city} {time} 20°，下雨"


if __name__ == "__main__":
    # 以 stdio 形式启动服务
    # stdio: 服务端和客户端在一台机器
    # streamable_http: 服务端和客户端在不同机器
    mcp.run(transport="stdio")

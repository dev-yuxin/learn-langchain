from datetime import date, datetime

from mcp.server.mcpserver import MCPServer
from pydantic import BaseModel, Field

# 创建MCP Server实例
mcp = MCPServer("MyMCPServer")


class WeatherArgs(BaseModel):
    city: str = Field(description="城市")
    time: date = Field(description="日期（精确到天）")


@mcp.tool()
def get_cur_time():
    """获取当前时间"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # noqa: DTZ005


@mcp.tool()
def get_weather(weather_input: WeatherArgs):
    """获取指定城市某天的天气"""
    return f"{weather_input.city} {weather_input.time} 20度，小雨"


# 启动 Server
if __name__ == "__main__":
    mcp.run(transport="stdio")  # 以Stdio方式运行

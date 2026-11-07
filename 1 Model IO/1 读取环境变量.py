import os

from dotenv import load_dotenv

# os.getenv() 默认优先读取系统环境变量 > .env文件
# load_dotenv(override=True) 使得 .env 的配置覆盖系统的环境变量
load_dotenv(override=True)

api_key = os.getenv("bailian_api_key")
base_url = os.getenv("bailian_base_url")
print(api_key, base_url)

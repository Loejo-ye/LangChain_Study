import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
# 加载 .env 文件里面全部环境变量
load_dotenv("../llm.env")

llm = ChatOpenAI(
    model="deepseek-v4-flash",
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    temperature=0
)

import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
# 加载 .env 文件里面全部环境变量
load_dotenv("../llm.env")


# DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "OPENAI_API_KEY")
def get_deepseek_llm(model: str = "deepseek-v4-flash", temperature: float = 0.0) -> ChatOpenAI:
    return ChatOpenAI(
        model=model,
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
        temperature=temperature
    )

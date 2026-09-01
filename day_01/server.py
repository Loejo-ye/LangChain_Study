import os
from typing import List
from fastapi import FastAPI
from langserve import add_routes
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
# 加载 .env 文件里面全部环境变量
load_dotenv("llm.env")


class Comma(BaseOutputParser[List[str]]):
    def parse(self, text: str) -> List[str]:
        return text.strip().replace("，", ",").split(",")


template = """
你是一个能生成以逗号分隔的列表的助手，用户会传入一个类别，你应该生成类别下的5个对象，并以逗号分隔的形式返回。
只返回以逗号分隔的内容。
不要包含其他内容。
"""
human_prompt = "{text}"
chat_prompt = ChatPromptTemplate([
    ("system", template),
    ("human", human_prompt)
])
llm = ChatOpenAI(
    model="deepseek-v4-flash",
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    temperature=0.9
)
# 流水线：提示词模板 → 大模型 → 自定义解析器Comma
# 前一个输出自动作为后一个的输入
first_chain = chat_prompt | llm | Comma()

# 定义应用,创建FastAPI应用实例
app = FastAPI(
    title="第一个LangChain应用",         # 服务标题，会显示在自动文档页面
    version="0.0.1",                   # 版本号
    description="LangChain应用接口"      # 服务描述
)
# 添加路由
# LangServer会自动生成/first_app/playground 调试画面
# add_routes：LangServe函数，把first_chain这条链挂载到web服务
# path="/first_app"：接口的基础路由前缀
# 自动生成接口：
# /first_app/invoke    一次性完整调用
# /first_app/stream    流式输出调用
# /first_app/playground 可视化调试网页，可以在浏览器直接测试
add_routes(app, first_chain, path="/first_app")
if __name__ == '__main__':
    # 导入uvicorn，ASGI服务器，用来运行FastAPI服务
    import uvicorn
    # 启动服务，监听本机8000 端口
    uvicorn.run(app, host="localhost", port=8000)

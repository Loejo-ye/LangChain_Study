import os
from typing import List
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
# 加载 .env 文件里面全部环境变量
load_dotenv("llm.env")


class CommaSeparatedListOutputParser(BaseOutputParser[List[str]]):
    def parse(self, text: str) -> List[str]:
        return text.strip().split(",")


# system系统提示词：设定AI角色，要求只返回逗号分隔的5个对象，不能输出多余文字
template = """
你是一个能生成以逗号分隔的列表的助手，用户会传入一个类别，你应该生成类别下的5个对象，并以逗号分隔的形式返回。只返回以逗号分隔的内容。
不要包含其他内容。
"""
# human用户消息模板，占位符{text}接收用户传入的类别
human_template = "{text}"
# 创建聊天提示模板，传入消息列表
# ("system", template) 系统角色；("human", human_prompt) 用户输入
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", template),
    ("human", human_template)
])
llm = ChatOpenAI(
    model="deepseek-v4-flash",
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    temperature=0.9
)
# 组装链
chain = chat_prompt | llm | CommaSeparatedListOutputParser()
if __name__ == '__main__':
    response = chain.invoke({"text": "植物"})
    print(response)

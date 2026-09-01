import os
import json
from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
# 加载 .env 文件里面全部环境变量
load_dotenv("llm.env")


class PersonInfo(BaseModel):
    """人物信息结构定义"""
    name: str = Field(description="姓名")
    occupation: str = Field(description="职业")
    fun_fact: str = Field(description="一个关于该人物的趣事")


template = """你是一个信息提取助手。请根据以下用户信息，生成一段自我介绍
用户信息如下:
{name}
{occupation}
{fun_fact}
请确保输出包含以上所有信息
"""
# 创建模板实例
person_info_prompt = ChatPromptTemplate.from_template(template)
llm = ChatOpenAI(
    model="deepseek-v4-flash",
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    temperature=0.9
)
if __name__ == '__main__':
    # 构造输入数据(pydantic自动校验字段)
    input_data = PersonInfo(
        name="张三",
        occupation="软件开发工程师",
        fun_fact="喜欢攀岩"
    )
    chain = person_info_prompt | llm
    # 将pydantic对象转为字典传入，或者直接传关键字参数
    response = chain.invoke({
        "name": input_data.name,
        "occupation": input_data.occupation,
        "fun_fact": input_data.fun_fact
    })
    print(response.content)

"""
结合 Pydantic 与自定义 OutputParser 的智能简历关键信息提取系统
实验目的：深入理解 BaseOutputParser 的自定义实现，并学会使用 Pydantic 定义结构化数据模型，解决大模型输出非结构化文本难以直接被程序解析的问题。
实验内容：
1、使用 Pydantic 定义一个 ResumeInfo 数据类，包含姓名、最高学历、工作年限、核心技能列表（List[str]）等字段。
2、继承 BaseOutputParser 自定义一个解析器，将大模型返回的特定文本格式（如 JSON 或特定分隔符文本）解析并转换为 Python 字典或对象。
3、输入一段非结构化的求职者自我介绍/简历文本，通过模型提取并自动输出标准的结构化数据。
"""
import os
from typing import List
from pydantic import BaseModel, Field
# 导入输出解析器基类，我们要继承它实现自定义解析器
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
# 加载 .env 文件里面全部环境变量
load_dotenv("llm.env")


class ResumeInfo(BaseModel):
    name: str = Field(description="姓名")
    xue_li: str = Field(description="最高学历")
    work_year: int = Field(description="工作年限")
    work: List[str] = Field(description="核心技能")


# 实例化解析器
parser = PydanticOutputParser(pydantic_object=ResumeInfo)

template = """
你是一个智能简历关键信息提取系统。
用户会给出一段文本信息，提取并自动输出标准的结构化数据
{format_instructions}
"""

# text = "{text}"
prompt = ChatPromptTemplate.from_messages([
    ("system", template),
    ("human", "简历内容：\n{text}")
]).partial(format_instructions=parser.get_format_instructions())

llm = ChatOpenAI(
    model="deepseek-v4-flash",
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    temperature=0
)
chain = prompt | llm | parser
if __name__ == '__main__':
    resume_text = ("我叫李明，本科学历，拥有3年相关行业工作经验，熟练掌握项目统筹与需求落地，具备扎实的数据分析能力，擅长跨部门沟通协作，能够高效推进业务任务落地执行。"
                   )
    try:
        response = chain.invoke({"text": resume_text})
        print(response)
        print(f"姓名：{response.name}")
        print(f"学历：{response.xue_li}")
        print(f"工作年限：{response.work_year}")
        print(f"核心技能：{response.work}")
    except Exception as e:
        print(f"调用出错：{e}")

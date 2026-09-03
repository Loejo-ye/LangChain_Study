# 自定义代理工具
import json
from typing import Optional
# pydantic数据校验
from pydantic import BaseModel, Field
from langchain_core.tools import tool
# MessagesPlaceholder草稿
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent

from config import get_deepseek_llm


class StudentQuery(BaseModel):
    # 参数校验模型
    student_id: str = Field(description="学生学号，例如：STU2026001")
    course_name: Optional[str] = Field(default=None, description="课程名称，如'人工智能概论'")


@tool("query_student_grade", args_schema=StudentQuery)
def query_student_grade(student_id: str, course_name: Optional[str] = None) -> str:
    """根据学生学号和课程名称查询成绩信息"""
    mock_db = {
        "STU2026001": {"姓名": "张三", "人工智能概论": 92, "数据结构": 88},
        "STU2026002": {"姓名": "李四", "人工智能概论": 85, "数据结构": 95},
    }

    # 根据学号从模拟数据库取出学生信息
    student_info = mock_db.get(student_id)
    if not student_info:
        return f"未查找到学号为{student_id}的学生数据。"

    if course_name:
        grade = student_info.get(course_name, "未修读此课程")
        return f"学生:{student_info['姓名']} {student_id}, 课程:{course_name}, 成绩:{grade}"

    return f"学生:{student_info['姓名']} {student_id}，全部成绩:{json.dumps(student_info, ensure_ascii=False)}"


def run():
    llm = get_deepseek_llm()
    tools = [query_student_grade]
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个校园学籍管理智能助手，请调用工具查询信息。"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    response = agent_executor.invoke({
        "input": "请帮我查一下学号STU2026001的学生在数据结构这门课考了多少分？"
    })
    print("\n代理输出：", response["output"])


if __name__ == '__main__':
    run()

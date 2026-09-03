# 通过工具增强实现文本大模型调度视觉识别与图标重构，完成跨模态交互
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent

from config import get_deepseek_llm


# 定义图像分析工具的入参校验模型
class ImageOCRInput(BaseModel):
    image_path_or_url: str = Field(description="图像文件路径或网络URL")


# 注册自定义定义工具，工具名称
@tool("analyze_image_content", args_schema=ImageOCRInput)
def analyze_image_content(image_path_or_url: str) -> str:
    """多模态图像分析工具：对传入的图片进行视觉内容识别，文字提取与场景描述"""
    return f"[图像视觉分析结果]：该图像包含一个折线图，展示了2026年第一季度AI模型调用量趋势；1月120万次，2月180万次，3月299万次"


# 第二个工具：图标生成工具，无Pydantic模型
@tool("generate_report_chart")
def generate_report_chart(chat_title: str, labels_json: str, values_json: str) -> str:
    """根据数据生成可视化图表工具"""
    return f"[图表生成成功]：已成功为：'{chat_title}'生成可视化数据报表，包含数据点：{labels_json} -> {values_json}"


@tool
def multiply_numbers(a: float, b: float) -> str:
    """计算两个数字的乘积"""
    return f"{a} * {b} = {a * b}"


def run():
    llm = get_deepseek_llm()
    tools = [analyze_image_content, generate_report_chart, multiply_numbers]

    # 给代理下达多模态任务执行策略：先读图，再计算，最后生成报表
    system_prompt = """
你是一个高级多模态智能代理。你能够协调调用图像分析工具、数据处理工具和报表生成工具来完成复杂的多模态交互任务。
当用户提供图像地址时，你应该先使用 analyze_image_content 工具提取图文信息，再基于提取到的信息完成后续的计算、总结或重新绘制报表等指令。
"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    # 用户多模态任务指令：传入图片链接、要求识图、求和、重绘图表
    user_query = "请分析这张图表图片https://example.com/charts/q1_usage.png，计算这三个月的总调用量，并把数据重新整理生成一份图标报告。"

    # 启动代理执行任务
    response = agent_executor.invoke({"input": user_query})
    print(response["output"])


if __name__ == '__main__':
    run()
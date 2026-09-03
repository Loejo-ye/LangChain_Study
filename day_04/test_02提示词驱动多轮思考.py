import json
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from config import get_deepseek_llm


# 1、知识库查询
@tool
def search_knowledge_base(query: str) -> str:
    """在内部知识库中搜索相关内容"""
    kb = {
        "LangChain": "LangChain是一个用于开发由语言模型驱动的应用框架。",
        "DeepSeek": "DeepSeek是一家专注于通用人工智能底座模型研发的高科技公司。",
        "ReAct": "ReAct是一种让大语言模型通过思考、行动和观察不断循环来完成任务的Agent架构。",
    }
    # 遍历知识库，不区分大小写匹配关键词
    for key, value in kb.items():
        if key.lower() in query.lower():
            return value
    return "知识库中没有相关记录"


@tool
def multiply_numbers(input_data: str) -> str:
    """计算两个数字的乘积"""
    try:
        data = json.loads(input_data)
        a = float(data["a"])
        b = float(data["b"])
        result = a * b
        return f"{a} x {b} = {result}"
    except json.JSONDecodeError:
        return (
            "参数格式错误。"
            "请输入JSON格式，例如："
            '{"a": 12, "b": 12}'
        )
    except KeyError:
        return (
            "缺少参数。"
            "请输入a和b"
        )
    except Exception as e:
        return f"计算失败: {str(e)}"


def run():
    llm = get_deepseek_llm()
    tools = [search_knowledge_base, multiply_numbers]
    for tool_item in tools:
        print(" -", tool_item.name)

    react_template = """
    尽你所能回答以下问题：
    你可以使用以下工具：
    {tools}
    工具名称：
    {tool_names}
    请严格按照以下格式进行思考和行动
    Question: 用户问题
    Thought: 我应该思考下一步该做什么
    Action: 要使用的工具名称；必须是[{tool_names}]中的一个
    Action Input: 工具输入
    Observation: 工具返回的结果
    可以重复Thought Action Action Input Observation
    直到得到最终答案
    最后必须使用：
    Thought: 我现在知道最终答案了
    Final Answer: 对用户问题的最终回答
    
    现在开始：
    Question: {input}
    Thought: {agent_scratchpad}
    """
    prompt = ChatPromptTemplate.from_template(
        react_template
    )
    from langchain_classic.agents import AgentExecutor, create_react_agent
    # 原始 ReAct 提示词版本 agent
    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=prompt
    )

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,                   # 详细日志
        handle_parsing_errors=True,     # 解析容错开关
        max_iterations=10,              # 最大循环轮次
    )

    question = (
        "请查一下什么是Deepseek？"
        "顺便算一下123456乘987654"
    )
    print(question)

    try:
        response = agent_executor.invoke(
            {
                "input": question,
            }
        )
        print(response["output"])
    except Exception as e:
        print(e)


if __name__ == '__main__':
    run()

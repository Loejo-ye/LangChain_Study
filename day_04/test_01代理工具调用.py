# @tool 是LangChain提供的快捷装饰器，将普通函数自动转换成Agent可识别的调用工具
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# 代理执行器，负责循环调度；大模型思考->调用工具->带回工具->生成最终答案
# create_tool_calling_agent：快速创建[工具调用型的智能代理]
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from config import get_deepseek_llm


@tool
def get_current_weather(city: str) -> str:
    """获取指定城市的实时天气信息"""
    weather_data = {
        "北京": "晴朗，气温22℃，微风",
        "上海": "小雨，气温19℃，东风3级",
        "广州": "多云，气温25℃，微风",
        "深圳": "晴朗，气温27℃，微风",
    }
    return weather_data.get(city, f"未查询到{city}的天气数据，默认提示：晴天，25℃")


@tool
def multiply_numbers(a: float, b: float) -> str:
    """计算两个数字的乘积"""
    return f"{a} * {b} = {a * b}"


def run():
    llm = get_deepseek_llm()
    # 将上面定义好的工具放入列表，传给Agent，代理拥有了这两项能力
    tools = [get_current_weather, multiply_numbers]

    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个聪明的助手，可以调用工具帮助用户解答问题。"),
        # 历史对话，是一堆消息对象，不是字符串，所以要用MessagesPlaceholder
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{input}"),
        # agent_scratchpad:代理的临时草稿区，LangChain工具调用代理必须的占位符
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    # 创建 Tool Calling Agent
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)   # verbose详细日志
    response = agent_executor.invoke({"input": "请问现在广州天气如何？另外，53乘以45等于多少？"})
    print("\n【代理输出】：", response["output"])


if __name__ == '__main__':
    run()

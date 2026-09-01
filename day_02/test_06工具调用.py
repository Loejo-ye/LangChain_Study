# @tool装饰器，用来将普通的python函数标记为Langchain可调用的工具
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage
from config import llm


# 创建计算器工具
@tool
def calculator(expression: str) -> str:
    """计算数学表达式"""
    try:
        # eval执行字符串形式的数学表达式
        # 第二个参数{}清空全局命名空间，关闭大部分内置函数，降低安全风险
        result = eval(expression,
                      {'__builtins__': {}})
        return str(result)
    except Exception as e:
        return f"计算错误:{e}"


@tool
def get_weather(city: str)-> str:
    """查询城市天气"""
    # 模拟本地天气字典数据库
    weather_data = {
        "北京": "晴天，25℃",
        "上海": "晴天，28℃",
        "广州": "晴天，32℃",
        "贵阳": "晴天，22℃",
        "深圳": "晴天，33℃"
    }
    return weather_data.get(
        city, "暂无天气数据"
    )


# 工具列表，存放所有可供模型使用的工具
tools = [
    calculator,
    get_weather
]
# bind_tools:给大模型绑定工具能力
llm_with_tools = llm.bind_tools(tools)
print("=" * 70)
print("实验七:ToolChain工具调用")
print("=" * 70)

questions =[
    "请计算1234乘以5678。"
    "请查询深圳的天气。"
]
for question in questions:
    print("\n用户" + question)
    # 调用绑定工具后的模型，模型会判断是否需要调用工具
    response = llm_with_tools.invoke(question)
    print("\n模型返回:")
    print(response)
    # tool_calls属性:模型输出的工具调用信息(工具名，传入参数)
    print("\n工具调用")
    print(response.tool_calls)

    # 执行工具调用，获取工具返回结果
    tool_messages = []
    for tool_call in response.tool_calls:
        # 根据工具名称找到对应的函数
        selected_tool = {tool.name: tool for tool in tools}[tool_call["name"]]
        # 执行工具，传入参数
        tool_output = selected_tool.invoke(tool_call["args"])
        # 包装成ToolMessage，需要带上tool_call的id
        tool_messages.append(ToolMessage(tool_output, tool_call_id=tool_call["id"]))

    # 把工具返回结果再次交给LLM，生成最终回答
    final_resp = llm_with_tools.invoke([response, *tool_messages])

    print("\n✅最终回答：")
    print(final_resp.content)
    print("-" * 70)






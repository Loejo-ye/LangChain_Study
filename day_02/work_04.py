"""
基于 LangChain 工具调用的智能电商客服助手开发
实验目标：
学习如何利用 LangChain 将自定义的 Python 业务工具绑定到大语言模型中，使大模型具备查询外部业务数据和计算的能力。
实验任务要求：
定义业务工具： 使用 @tool 装饰器编写至少两个全新的业务函数：
get_order_status(order_id: str)：根据订单号查询订单物流状态（示例数据可内置字典，如发货状态、快递单号等）。
calculate_member_discount(original_price: float, member_level: str)：根据商品原价和会员等级（如普通、黄金、钻石）计算折后最终价格。
模型绑定与调用： 初始化大语言模型，并使用 llm.bind_tools() 将上述工具绑定到模型上。
交互验证： 编写测试用例，向模型询问包含订单查询和价格计算的问题，并打印输出模型的响应结果及生成的 tool_calls 工具调用指令。
"""
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage
from config import llm

# 订单列表：订单号：发货状态
order_dict = {
    "1001": "已发货",
    "1002": "未发货",
    "1003": "已发货",
}
# for o_id in order_dict:
#     print(o_id)


@tool
def get_order_status(order_id: str):
    """根据订单号查询订单物流状态"""
    try:
        for o_id in order_dict:
            if o_id == order_id:
                return order_dict[o_id]
    except Exception as e:
        return f"错误:{e}"


# 会员等级（如普通、黄金、钻石）
member_discount = {
    "普通": 1.0,
    "黄金": 0.8,
    "钻石": 0.6
}


@tool
def calculate_member_discount(original_price: float, member_level: str):
    """根据商品原价和会员等级（如普通、黄金、钻石）计算折后最终价格"""
    discount = member_discount.get(member_level, 1.0)
    return round(original_price*discount, 2)


# 工具列表，存放所有可供模型使用的工具
tools = [
    get_order_status,
    calculate_member_discount
]

# bind_tools:给大模型绑定工具能力
llm_with_tools = llm.bind_tools(tools)

# 向模型询问包含订单查询和价格计算的问题，
# 并打印输出模型的响应结果及生成的 tool_calls 工具调用指令
questions = [
    "查询订单1001的物流状态",
    "计算订单1002的价格,原价格是100，我的会员等级是钻石"  # 原价格：{original_price}，会员等级：{member_level}
]

for question in questions:
    print("\n用户询问：" + question)
    response = llm_with_tools.invoke(question)
    print("\n模型返回:")
    print(response)
    print("\n工具调用:")
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

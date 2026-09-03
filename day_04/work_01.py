"""
基于 ReAct 思考链的智能物流与客服调度代理
场景需求
在电商物流客服场景中，用户常提出包含多步操作的复杂问题（如：“帮我查询订单 SF9981 的快递到达何处，如果更改为航空加急，额外支付 15% 费用后的总运费是多少？”）。代理需具备逻辑推理与自主决策能力，先查询基础运单状态，再调用计算逻辑，最终给出完整的合规解答。
核心技术要点
1、理解并实现基础的工具调用机制（Tool Calling）。
2、使用 Prompt 模板显式定义 ReAct（Thought - Action - Action Input - Observation）思考链结构。
3、使用 AgentExecutor 驱动大语言模型完成多轮“思考-行动”迭代，并建立工具调用的容错与降级机制。
实验任务
1、自定义物流状态查询函数与运费加价计算工具。
2、设计包含 tools、tool_names 及 agent_scratchpad 的 ReAct 提示词模板。
3、构建 Agent 并测试复杂多步骤任务的推理与执行过程。
"""
import json
from langchain_core.tools import tool
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import ChatPromptTemplate
from config import get_deepseek_llm


@tool
def check_order(input_data: str) -> str:
    """查询订单位置，Action Input传入JSON字符串 {"order_id":"订单编号"}"""
    import json
    try:
        data = json.loads(input_data)
        order_id = data["order_id"]
    except Exception:
        return f"订单解析失败，请传入JSON格式，例如 {{\"order_id\":\"SF9981\"}}"

    order_list = {
        "SF9981": "上海",
        "SF9982": "北京"
    }
    position = order_list.get(order_id.upper())
    if position:
        return f"订单{order_id}当前位置：{position}"
    return f"未查询到订单{order_id}的快递"


@tool
def calc_express_fee(input_data: str) -> str:
    """计算需要加急后的价格，输入JSON字符串{"a":原运费}"""
    import re
    try:
        data = json.loads(input_data)
        a = float(data["a"])
    except Exception:
        # 兼容大模型输出中文文本的情况，提取数字
        match = re.search(r"(\d+\.?\d*)", input_data)
        if not match:
            return "错误：无法解析运费，请传入JSON格式 {\"a\":数值}"
        a = float(match.group(1))
    result = round(a * 1.15, 2)
    return f"加急后的价格是{result}"


def run():
    llm = get_deepseek_llm()
    tools = [check_order, calc_express_fee]
    result_prompt = """
具备逻辑推理与自主决策能力，先查询基础运单状态，再调用计算逻辑，最终给出完整的合规解答来回答以下问题：
你可以使用以下工具：
{tools}
工具名称：
{tool_names}
请严格按照以下格式进行思考和行动:
如果工具需要JSON输入，Action Input中**只输出纯粹JSON字符串，不能附带中文说明，不要加任何多余文字**。
不要重复调用已经拿到结果的工具。
Question: 用户问题
Thought: 我应该思考下一步该做什么
Action: 要使用的工具名称；必须是[{tool_names}]中的一个
Action Input: 工具输入
Observation: 工具返回的结果
可以重复Thought-Action-Action Input-Observation
直到得到最终答案
最后必须使用：
Thought: 我现在知道最终答案了
Final Answer: 对用户问题的最终回答

现在开始：
Question: {input}
Thought: {agent_scratchpad}
"""
    prompt = ChatPromptTemplate.from_template(
        result_prompt
    )
    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,                   # 详细日志
        handle_parsing_errors=True,     # 解析容错开关
        max_iterations=10,              # 最大循环轮次
    )

    question = (
        "请查一下订单号SF9981的快递在哪"
        "另外，快递原运费100，我要更改为航空加急，总运费是多少？"
    )

    try:
        response = agent_executor.invoke(
            {
                "input": question,
            }
        )
        print("\n【输出结果】", response["output"])
    except Exception as e:
        print(e)


if __name__ == '__main__':
    run()













"""
基于 RunnableLambda 的用户反馈多维度分析流水线
实验目标：
学习使用 RunnableLambda 将自定义 Python 文本处理逻辑包装为 LangChain 可执行组件（Runnable），并将其组合构成多任务协同运行的分析流水线。
实验任务要求：
1、编写单项分析逻辑：
(1)编写函数一 analyze_sentiment(text)：调用大模型分析用户评价的情感倾向（正面/负面/中性）并给出置信度。
(2)编写函数二 extract_action_items(text)：调用大模型从评价中提取产品改进建议或待办事项。
2、构建 Runnable 组件： 使用 RunnableLambda 将上述两个分析函数分别封装为标准 LangChain 组件 sentiment_chain 和 action_chain。
3、组装复合 Chain： 编写主处理函数 feedback_analysis_pipeline(text)，顺序/并行调用上述两个组件，并汇总返回包含 sentiment 和 action_items 的字典结构。
真实样本测试： 传入一段真实的软件产品用户评语，测试分析流水线并格式化打印最终的分析结果。
"""
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config import llm


# 函数1
def analyze_sentiment(text):
    prompt = f"""
    根据下面文本分析用户评价的情感倾向（正面/负面/中性）并给出置信度.
    文本：{text}
    """
    response = llm.invoke(prompt)
    return response.content


# 函数2
def extract_action_items(text):
    prompt = f"""
    从评价中提取产品改进建议或待办事项。
    评价：{text}
    """
    response = llm.invoke(prompt)
    return response.content


# 构建 Runnable 组件：
# 使用 RunnableLambda将上述两个分析函数分别封装为标准 LangChain 组件 sentiment_chain 和 action_chain。
sentiment_chain = RunnableLambda(analyze_sentiment)
action_chain = RunnableLambda(extract_action_items)


# 组装复合 Chain： 编写主处理函数
def feedback_analysis_pipeline(text):
    sentiment = sentiment_chain.invoke(text)
    action = action_chain.invoke(text)
    # 返回包含sentiment和action_items的字典结构
    return {
        "sentiment": sentiment,
        "action": action
    }


if __name__ == '__main__':
    text = input("请输入评价:")
    result = feedback_analysis_pipeline(text)
    print("\n情感倾向:")
    print(result["sentiment"])
    print("\n改进建议或待办事项:")
    print(result["action"])







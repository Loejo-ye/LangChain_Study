"""
基于 RunnableParallel 与 RunnableBranch 的多任务智能客服分流系统
实验目的：掌握复杂 Runnable 的编排方式，理解使用 RunnableParallel 进行多维并行特征提取以及使用 RunnableBranch 实现基于规则的动态路由。
实验任务：
1、并行特征分析链：使用 RunnableParallel 构建多任务处理链，同时对输入的客服工单进行“关键要素提取”和“一句话摘要”的并行分析。
2、条件路由分流链：编写路由断言函数（判断工单中是否包含“代码”、“数据库”、“报错”等技术关键字），结合 RunnableBranch 将工单动态路由至“技术专家”或“普通售后客服”的不同处理链中。
综合集成与测试：编写主处理函数，输入任意客户咨询文本，同步返回并行分析元数据（关键词与摘要）以及最终路由分配后的专业解答。
"""
from langchain_core.runnables import (
    RunnableBranch, RunnableParallel
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from config import llm

# 并行处理
keyword_prompt = ChatPromptTemplate.from_template(
    "从下面的问题中提取3个关键词，只输出关键词:\n{question}"
)
summary_prompt = ChatPromptTemplate.from_template(
    "将下面的问题总结成一句话:\n{question}"
)

keyword_chain = keyword_prompt | llm | StrOutputParser()
summary_chain = summary_prompt | llm | StrOutputParser()
parallel_chain = RunnableParallel(
    keywords=keyword_chain,
    summary=summary_chain
)

# 分支处理
technical_chain = (
    ChatPromptTemplate.from_template(
        "你是一名技术专家，请回答:{question}"
    )
    | llm
    | StrOutputParser()
)
normal_chain = (
    ChatPromptTemplate.from_template(
        "你是一名普通售后客服，请回答:{question}"
    )
    | llm
    | StrOutputParser()
)


def is_technical(x):
    question = x["question"]
    keywords = [
        "代码", "数据库", "报错"
    ]
    return any(k in question for k in keywords)


branch_chain = RunnableBranch(
    (is_technical, technical_chain),
    normal_chain
)

# 组装完整LCEL链
# 一次invoke，同时得到analysis(关键词+摘要) 和 answer(路由后的回答)
full_chain = RunnableParallel(
    analysis=parallel_chain,
    answer=branch_chain
)


# def process(question):
#     # 并行链执行:提取关键词 +生成摘要
#     analysis = parallel_chain.invoke({
#         "question": question
#     })
#     answer = branch_chain.invoke({
#         "question": question
#     })
#     return {
#         "analysis": analysis,
#         "answer": answer
#     }


if __name__ == '__main__':
    # result = process(input("请输入文本:"))
    user_input = input("请输入文本:")
    result = full_chain.invoke({"question": user_input})
    # print(result)
    print("=" * 10 + "关键词" + "=" * 10)
    print(result["analysis"]["keywords"])
    print("\n" + "=" * 10 + "摘要" + "=" * 10)
    print(result["analysis"]["summary"])
    print("\n" + "=" * 10 + "回答" + "=" * 10)
    print(result["answer"])


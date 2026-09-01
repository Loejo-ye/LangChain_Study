# RunnableParallel:并行执行多条链路，同时跑多个任务
# RunnableBranch：条件分支，根据判断选择执行哪一条链路
# RunnableLambda：把普通Python函数包装成可加入LCEL链路的节点
from langchain_core.runnables import (
    RunnableLambda, RunnableBranch, RunnableParallel
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config import llm

# 1.并行处理
# 提取关键词的提示词模板
keyword_prompt = ChatPromptTemplate.from_template(
    "从下面的问题中提取3个关键词，只输出关键词:\n{question}"
)
# 问题摘要总结的提示模板
summary_prompt = ChatPromptTemplate.from_template(
    "将下面的问题总结成一句话:\n{question}"
)
# 构建关键词提取子链
keyword_chain = keyword_prompt | llm | StrOutputParser()
# 构建问题摘要子链
summary_chain = summary_prompt | llm | StrOutputParser()
parallel_chain = RunnableParallel(  # 两条子链同时执行
    keywords=keyword_chain,
    summary=summary_chain
)  # {"keywords":关键词结果，"summary":摘要结果}

# 分支处理
# 技术问题分支链路:设定角色为技术专家
technical_chain = (
    ChatPromptTemplate.from_template(
        "你是一名技术专家，请回答:{question}"
    )
    | llm
    | StrOutputParser()
)
# 普通问题分支链路:设定角色为智能客服
normal_chain = (
    ChatPromptTemplate.from_template(
        "你是一名智能客服，请回答:{question}"
    )
    | llm
    | StrOutputParser()
)


def is_technical(x):
    question = x["question"]
    keywords = [
    "Python", "Langchain", "数据库", "API", "模型", "CPU", "代码"
    ]
    return any(k in question for k in keywords)


branch_chain = RunnableBranch(
    (is_technical,technical_chain),
    normal_chain
)


def process(question):
    # 并行链执行:提取关键词 +生成摘要
    analysis = parallel_chain.invoke({
        "question": question
    })
    answer = branch_chain.invoke({
        "question": question
    })
    return {
        "analysis": analysis,
        "answer": answer
    }


result = process(
    "LangChain中的Runnable是什么"
)
print("="*10 + "关键词" + "="*10)
print(result["analysis"]["keywords"])
print("\n" + "="*10 + "摘要" + "="*10)
print(result["analysis"]["summary"])
print("\n" + "="*10 + "回答" + "="*10)
print(result["answer"])









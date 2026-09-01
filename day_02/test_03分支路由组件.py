# RunnableLambda：把普通Python函数包装成可加入LCEL链路的节点
# RunnablePassthrough：原样把输入数据往下传递
from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableLambda
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config import llm


# 自定义文本处理函数
# 注意指定text变量类型
def clean_text(text: str):
    return text.strip().lower()


# 将普通Python函数包装成Runnable节点，放到LCEL管道链中执行
clean_node = RunnableLambda(clean_text)
prompt = ChatPromptTemplate.from_template(
    """
    你是一名AI教师。
    原始问题：{original}
    处理后的问题：{cleaned}
    请回答用户问题
    """
)

# 构建Runnable链路
chain = (
    {
        "original": RunnablePassthrough(),
        "cleaned": clean_node
    }
    | prompt
    | llm
    | StrOutputParser()
)
question = "什么是LangChain"
result = chain.invoke(question)
print(result)






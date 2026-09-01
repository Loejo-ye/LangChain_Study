from langchain_core.prompts import ChatPromptTemplate
# 输出解析器模块导入字符串解析器
from langchain_core.output_parsers import StrOutputParser
from config import llm
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "你是一个人工智能专业教师，回答要准确、简洁。"
    ),
    (
        "human",
        "请解释以下什么是{topic}"
    )
])
# 创建输出解析器实例
# 将大模型返回的对象提取处理出里面的字符串内容
parser = StrOutputParser()
# 构建LCEL链式管道 |
chain = prompt | llm | parser
# 查看当前Chain接收的输入数据结构(Schema)
# input_schema会告诉我们这条链需要传入哪些参数，参数类型是什么
print("输入Schema")
print(chain.input_schema.schema())
# 调用链执行推理，invoke是同步调用方法
# 传入字典
result = chain.invoke({
    "topic": "LangChain",
})
print("\n模型回答:")
print(result)

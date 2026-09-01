import asyncio
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config import llm
prompt = ChatPromptTemplate.from_template(
    "请用100~300字以内介绍：{topic}"
)
chain = prompt | llm | StrOutputParser()
print("=" * 60)
print("1.invoke")
print("=" * 60)

# 1.invoke同步执行链,传入字典参数填充topic占位符
result = chain.invoke({
    "topic": "人工智能"
})
print(result)

# 2.同步流式输出，一块一块接收模型返回内容
print("\n" + "=" * 60)
print("2.Stream")
print("\n" + "=" * 60)
# 遍历流式返回的数据块chunk
for chunk in chain.stream({"topic": "大语言模型"}):
    print(chunk, end="", flush=True)
print()
# 3.batch:批量同步调用，一次性处理多组输入任务
print("\n" + "=" * 60)
print("3.batch")
print("\n" + "=" * 60)
# 定义一批需要处理的输入列表，每一项都是一条任务参数
topics = [
    {"topic": "Python"},
    {"topic": "LangChain"},
    {"topic": "RAG"},
    {"topic": "智能体"}
]
results = chain.batch(topics)
for i, result in enumerate(results):
    print(f"\n问题{i+1}:")
    print(result)


# 4.stream:异步流式输出
async def async_stream():
    print("\n" + "=" * 60)
    print("4.astream")
    print("\n" + "=" * 60)

    async for chunk in chain.astream({"topic": "AI Agent"}):
        print(chunk, end="", flush=True)

    print()
# asyncio.run()启动异步时间循环，执行异步函数
asyncio.run(async_stream())

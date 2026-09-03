from langchain_openai import ChatOpenAI
# ConversationBufferMemory：LangChain经典的缓存式对话记忆组件，用来保存完整的聊天历史
from langchain_classic.memory import ConversationBufferMemory
# 最简单的多轮对话链，大模型+记忆，实现带上文记忆的聊天
from langchain_classic.chains import ConversationChain
from config import get_deepseek_llm

llm = get_deepseek_llm()

# 创建记忆
# 初始化对话缓存记忆对象，会把所有历史问答完整保存下来，每次对话全部传给大模型
memory = ConversationBufferMemory()

conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True,
)

print("第一轮")
print(
    conversation.predict(
        input="你好，我叫张三。"
    )
)

print("\n第二轮")
print(
    conversation.predict(
        input="我目前正在学习Python编程"
    )
)

print("\n第三轮：")
print(
    conversation.predict(
        input="我叫什么？我正在学习什么？"
    )
)

print("Memory中的内容")
# 读取内存里所存储的全部对话历史上下文。
print(memory.load_memory_variables({}))

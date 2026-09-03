from langchain_openai import ChatOpenAI
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains import ConversationChain
from config import get_deepseek_llm
import warnings
warnings.filterwarnings("ignore")

llm = get_deepseek_llm()

# 初始化滑动窗口记忆对象
memory = ConversationBufferMemory(k=2)

# 创建对话链，绑定大模型与滑动窗口记忆
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True,
)

questions = [
    "我叫张三。",
    "我来自贵州。",
    "我喜欢python。",
    "我正在学习LangChain。",
    "我准备学习RAG",
]

for question in questions:
    print("\n用户：", question)
    answer = conversation.predict(input=question)
    print("AI：", answer)

print(memory.load_memory_variables({}))

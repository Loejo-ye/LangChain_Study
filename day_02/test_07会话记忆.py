from langchain_openai import OpenAI
from langchain_core.prompts import ChatPromptTemplate
# 内存存储聊天历史，程序关闭数据就丢失
from langchain_core.chat_history import InMemoryChatMessageHistory
# 给普通链路添加上下文记忆能力
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from config import llm

# 构造带对话历史占位符的Prompt模板
prompt = ChatPromptTemplate.from_messages([
    ("system",
     """你是一名人工智能专业教师。
请使用简洁、准确的中文回答学生的问题。"""),
    (
        "placeholder",
        "{history}"
    ),
    (
        "human",
        "{question}"
    )
])

# 创建基础LCEL(无记忆)
chain = (prompt | llm | StrOutputParser())

# 会话历史存储器与读取函数
store = {}


def get_session_history(session_id):
    # 如果该会话编号不存在，则新建一条内存对话历史
    if session_id not in store:
        store[session_id] = (
            InMemoryChatMessageHistory()
        )
    return store[session_id]


# 包装成带记忆的对话链
chat_chain = RunnableWithMessageHistory(
    chain,                          # 基础无记忆链路
    get_session_history,            # 获取历史记录的回调函数
    input_messages_key="question",  # 用户输入字段名
    history_messages_key="history"
)

config = {
    "configurable": {
        "session_id": "student001"
    }
}

print("=" * 70)
print("实验七: 多轮对话Chain")
print("=" * 70)

# 连续三轮问题，上下文互相依赖
questions = [
    "什么是LangChain？",
    "它主要有什么功能？",
    "它和RAG有什么关系？"
]

for question in questions:
    print("\n" + "="*20)
    print("用户询问: " + question)
    # 调用带记忆的对话链，config传入会话标识
    result = chat_chain.invoke(
        {
            "question": question
        },
        config=config
    )
    print("\nAI回答:" + result)

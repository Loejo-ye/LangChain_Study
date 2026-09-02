from openai import OpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import os
from dotenv import load_dotenv
# 加载 .env 文件里面全部环境变量
load_dotenv("../llm.env")

# 创建大模型客户端实例
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)

# 加载向量嵌入模型
embeddings = HuggingFaceEmbeddings(
    model_name="D:/pycharm practive/LangChain/pythonProject/day_03/model/bge-small-zh-v1.5",  # 本地中文向量模型
    model_kwargs={
        "device": "cpu"  # 指定使用CPU运行模型
    },
    encode_kwargs={
        "normalize_embeddings": True,  # 归一化
    }
)

# 加载之前已经建好，持久化在磁盘的向量数据库
vector_store = Chroma(
    collection_name="tencent_cloud",
    embedding_function=embeddings,
    persist_directory="./vector_db/tencent"
)

# 检索
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 4
    }
)

question = input("\n请输入问题：")
docs = retriever.invoke(question)
print("检索到的知识")
for i, doc in enumerate(docs):
    print(f"\n【知识{i + 1}】")
    print("来源：", doc.metadata.get("filename"))
    print(doc.page_content)

# 将多条召回文档拼接成一段上下文Context
context = "\n\n".join(
    doc.page_content for doc in docs
)

# 构造RAG提示词,约束大模型只能基于知识库回答
prompt = f"""
你是一名企业知识库智能问答助手。
请严格根据下面提供的知识库内容回答用户问题。
【知识库】{context}
【用户问题】{question}
要求：
1、只能根据知识库回答。
2、不允许编造知识库不存在的信息。
3、如果知识库没有答案，请回答：
“根据当前知识库，无法找到相关信息。”
4、回答要简洁、准确
"""

response = client.chat.completions.create(
    model="deepseek-v4-flash",
    messages=[
        {
            "role": "system",
            "content": "你是一个专业的企业知识库问答助手"
        },
        {
            "role": "user",
            "content": prompt
        }
    ],
    temperature=0.2
)

print(response.choices[0].message.content)

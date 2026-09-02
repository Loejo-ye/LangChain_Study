# 把文字转化为向量操作
from langchain_huggingface import HuggingFaceEmbeddings

model_name = "D:/pycharm practive/LangChain/pythonProject/day_03/model/bge-small-zh-v1.5"

embeddings = HuggingFaceEmbeddings(
    model_name=model_name,              # 本地模型路径
    model_kwargs={
        "device": "cpu"                 # 指定使用CPU运行模型
    },
    encode_kwargs={
        "normalize_embeddings": True,   # 归一化
    }
)

texts = [
    "腾讯云提供云服务器、对象存储、云数据库、人工智能等服务。",
    "腾讯云的对象存储主要用于保存图片、视频和文档等非结构化数据。",
    "Serverless是一种无需开发者管理服务器基础设施的计算方式。",
    "人工智能专业主要培养人工智能应用开发的相关技术人才。",
    "RAG可以通过检索外部知识库来增强大语言模型的回答能力。"
]

# 批量把多条文档文本转换为向量列表
vectors = embeddings.embed_documents(texts)
print("文本数量:", len(vectors))

# 打印第一条文本对应的向量数组，打印向量维度大小
print("向量维度:", len(vectors[0]))
print("\n第一个文本:")
print(texts[0])
print(vectors[0][:10])
query = "腾讯云有哪些服务?"
# 单独对一条查询语句生成向量
query_vector = embeddings.embed_query(query)

print("\n查询:")
print(query)
print("\n查询向量维度:")
print(len(query_vector))

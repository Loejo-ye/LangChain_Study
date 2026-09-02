# 导入HuqqingFace嵌入模型，用于生成文本向量
from langchain_huggingface import HuggingFaceEmbeddings
# 导入Chroma向量数据库
from langchain_chroma import Chroma

embeddings = HuggingFaceEmbeddings(
    model_name="D:/pycharm practive/LangChain/pythonProject/day_03/model/bge-small-zh-v1.5",              # 本地模型路径
    model_kwargs={
        "device": "cpu"                 # 指定使用CPU运行模型
    },
    encode_kwargs={
        "normalize_embeddings": True,   # 归一化
    }
)

# 读取磁盘上已经建好的向量数据库
vector_store = Chroma(
    collection_name="tencent_cloud",            # 向量集合名称，建库时定义的名字，必须一致
    embedding_function=embeddings,              # 指定用于检索的向量模型
    persist_directory="./vector db/tencent"     # 向量数据库在本地磁盘存放的文件夹路径
)

# 创建检索器
retriever = vector_store.as_retriever(
    search_type="similarity",   # 检索方式:找语义最相近的文档
    search_kwargs={
        "k": 3                  # 返回相似度最高的前3个文本块
    }
)

# 用户输入问题并执行检索
question = input("请输入问题:")
docs = retriever.invoke(question)   # invoke方法:将问题转为向量，在向量库中检索召回文档
print("\n==========================================")
print("检索结果")
print("==========================================")

for i, doc in enumerate(docs):
    print(f"\n---Top {i+1} ---")  # 输出排名序号，从1开始
    print(doc.page_content)       # 打印召回文本块的正文内容
    print("\nMetadata")
    print(doc.metadata)           # 打印召回文本块的元数据

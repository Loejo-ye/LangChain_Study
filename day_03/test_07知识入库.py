from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# 指定待读取的txt文档路径
DATA_DIR = Path("data/docs")
# 定义列表，用来保存全部读取完成的Document文档对象
documents = []

# 遍历文件夹下所有后缀为txt的文件
for file in DATA_DIR.glob("*txt"):
    print("加载:", file.name)
    # 实例化文本加载器，指定文件路径和utf‑8编码，防止中文乱码
    loader = TextLoader(
        str(file), encoding="utf-8"
    )
    # 读取单个txt文件
    docs = loader.load()
    # 遍历本次加载出来的文档，给元数据增加filename字段，记录来源文件名
    for doc in docs:
        doc.metadata["filename"] = file.name  # 记录来源文件名
    # extend：把当前文件的文档对象合并到总列表documents
    documents.extend(docs)
print("\n总文档数量:", len(documents))

# 创建文本分割器
splitter = RecursiveCharacterTextSplitter(
    # 每个文本块最大字符长度，150个字符
    chunk_size=150,
    # 块与块之间重叠字符数:30个字符
    chunk_overlap=30,
    separators=[    # 分割优先级列表，从上到下依次尝试分割
        "\n\n",
        "\n",
        "。",
        ".",
        "、",
        " ",
        "",
    ]
)
chunks = splitter.split_documents(documents)  # 执行分割，得到切分后的所有文本块
print("chunk数量:", len(chunks))

# 初始化向量嵌入模型
embeddings = HuggingFaceEmbeddings(
    model_name="D:/pycharm practive/LangChain/pythonProject/day_03/model/bge-small-zh-v1.5",              # 本地模型路径
    model_kwargs={
        "device": "cpu"                 # 指定使用CPU运行模型
    },
    encode_kwargs={
        "normalize_embeddings": True,   # 归一化
    }
)

# 创建并持久化Chroma向量库
vector_store = Chroma.from_documents(
    documents=chunks,                       # 需要存入向量库的文本切块
    embedding=embeddings,                   # 使用哪个向量模型生成向量
    collection_name="tencent_cloud",        # 向量集合名称，一个库可以存放多个不同集合
    persist_directory="./vector_db/tencent"
)

# 创建检索器
retriever = vector_store.as_retriever(
    search_type="similarity",   # 检索方式:找语义最相近的文档
    search_kwargs={
        "k": 5                  # 返回相似度最高的前5个文本块
    }
)

question = input("\n请输入问题:")
results = retriever.invoke(question)
for i, doc in enumerate(results):
    print(f"回答{i+1}")
    print("来源:", doc.metadata.get("filename"))
    print("内容:")
    print(doc.page_content)

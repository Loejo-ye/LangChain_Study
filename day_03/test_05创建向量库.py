# 导入路径工具类Path，用于处理文件路径
from pathlib import Path
# 导入文本加载器，用于读取txt纯文本文件
from langchain_community.document_loaders import TextLoader
# 导入递归字符文本分割器，用于把长文档切分成多个文本块chunk
from langchain_text_splitters import RecursiveCharacterTextSplitter
# 导入HuggingFace嵌入类，加载本地bge向量化模型，把文字转为向量
from langchain_huggingface import HuggingFaceEmbeddings
# 导入Chroma本地向量数据库，用来存储文本向量，支持磁盘持久化
from langchain_chroma import Chroma

# 指定待读取的txt文档路径
file_path = "data/docs/腾讯云介绍.txt"

# 实例化文本加载器，传入文件路径与读取编码
loader = TextLoader(
    file_path,
    encoding="utf-8"
)

# 读取文件，返回包含Document对象的列表
# Document对象包含page_content（文本内容）、metadata（元数据）
documents = loader.load()

# 创建文本分割器
text_splitter = RecursiveCharacterTextSplitter(
    # 每个文本块最大字符长度，150个字符
    chunk_size=150,
    # 块与块之间重叠字符数:30个字符
    chunk_overlap=30,
    # 分割符优先级，从上到下依次尝试切割，优先按换行、句号等语义边界切分
    separators=[
        "\n\n",
        "\n",
        "。",
        ".",
        "、",
        " ",
        "",
    ]
)

# 对加载好的文档进行切块，返回切块后的对象列表
chunks = text_splitter.split_documents(documents)

# 初始化本地中文Embedding向量模型
embeddings = HuggingFaceEmbeddings(
    # 本地模型路径
    model_name="D:/pycharm practive/LangChain/pythonProject/day_03/model/bge-small-zh-v1.5",
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
print("向量数据库创建成功!")
print("Chunk数量:", len(chunks))



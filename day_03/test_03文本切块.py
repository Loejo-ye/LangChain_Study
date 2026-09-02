from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

file_path = "data/docs/腾讯云介绍.txt"

loader = TextLoader(
    file_path,
    encoding="utf-8"
)
# 读取文件，返回包含Document对象的列表
documents = loader.load()
# 创建文本分割器
text_splitter = RecursiveCharacterTextSplitter(
    # 每个文本块最大字符长度，150个字符
    chunk_size=150,
    # 块与块之间重叠字符数:30个字符
    chunk_overlap=30,
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
print("原始文档数量:", len(documents))
print("切分后的chunk数量:", len(chunks))
for i, chunk in enumerate(chunks):
    print("\n=======================================")
    print(f"Chunk {i + 1}")
    print(chunk.page_content)
    print("\n Metadata")
    print(chunk.metadata)

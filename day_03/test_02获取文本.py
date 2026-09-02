from pathlib import Path
from langchain_community.document_loaders import TextLoader

DATA_DIR = Path("data/docs")

files = list(DATA_DIR.glob("*.txt"))
print("发现文档")

for file in files:
    print(file)
print("\n开始加载文档")
documents = []

for file in files:
    loader = TextLoader(str(file), encoding="utf-8")
    docs = loader.load()
    documents.extend(docs)  # 把本次读取的文档对象全部合并到documents总列表
print("\n文档数量:", len(documents))
for i, doc in enumerate(documents):
    print("\n=========================================")
    print("文档", i+1)
    print("\n=========================================")
    print("内容")
    print(doc.page_content[:300])
    print(doc.metadata)





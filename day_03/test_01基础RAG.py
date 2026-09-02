import os
from openai import OpenAI
from dotenv import load_dotenv
# 加载 .env 文件里面全部环境变量
load_dotenv("../llm.env")

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)

# 模拟知识库(本地文档库)
knowledge_base = [
    "腾讯云提供云服务器、对象存储、云数据库、人工智能等服务。",
    "腾讯云的对象存储主要用于保存图片、视频和文档等非结构化数据。",
    "Serverless是一种无需开发者管理服务器基础设施的计算方式。",
    "人工智能专业主要培养人工智能应用开发的相关技术人才。",
    "RAG可以通过检索外部知识库来增强大语言模型的回答能力。"
]

# 简单关键词检索模块(简易RAG召回)
question = input("请输入问题:")
# 创建空列表，用来存放检索后命中的知识库片段(召回结果
retrieved_doc = []

# 遍历知识库每一条文档
for doc in knowledge_base:
    # 逻辑:只要问题里任意一个词出现在文档文本里，就命中这条文档
    if any(word in doc for word in question):
        retrieved_doc.append(doc)
# 如果检索结果为空，一条文档都没有匹配到
if not retrieved_doc:
    retrieved_doc = knowledge_base[:2]

# 将召回得到的多条文档，用换行符拼接成一段完整的上下文文本
context = "\n".join(retrieved_doc)
# 构造提示词Prompt
prompt = f"""
你是一个知识库问答助手
请严格根据下面的知识库内容回答问题
知识库:{context}
用户问题:{question}
如果知识库中没有相关信息，请明确说明“知识库中没有找到相关信息”
不要编造知识库中不存在的内容。
"""
# 调用DeepSeek大模型聊天补全接口
response = client.chat.completions.create(
    model="deepseek-v4-flash",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ],
    temperature=0.2
)
print("="*50)
print("RAG回答")
# print(response)
# print("="*50)
# print(response.choices[0])
# print("="*50)
# print(response.choices[0].message)
# print("="*50)
print(response.choices[0].message.content)

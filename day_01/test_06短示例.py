import os
from langchain_core.prompts import(
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
# 加载 .env 文件里面全部环境变量
load_dotenv("llm.env")

# 定义示例的格式化模板
examples = [
    {"input": "1+1等于多少?", "output": "2"},
    {"input": "2+3等于多少?", "output": "5"},
    {"input": "3+1等于多少?", "output": "4"},
]
# 定义示例的格式化模板
# 告诉模型每个示例应该长什么样(人类提问，AI回答)
example_prompt = ChatPromptTemplate.from_messages([
    ("human", "{input}"),
    ("ai", "{output}"),
])
# 创建few-shot提示模板
# 将示例和格式化模板组合起来
few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)

# 构建完整的对话提示
# 包含:系统指令 +少样本示例 + 用户实际输入
final_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个数学的助手，请根据下面的示例格式回答问题。"),
    few_shot_prompt,
    ("human", "{input}"),  # 用户的实际输入变量
])
# 调用Deepseek模型
model = ChatOpenAI(
    model="deepseek-v4-flash",
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    temperature=0  # 数学题通常设低温度，保证准确性
)
chain = final_prompt | model

response = chain.invoke({"input": "28-12等于多少?"})
print(f"用户输入:28-12等于多少?")
print(f"模型回答:{response.content}")

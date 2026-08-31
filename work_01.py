"""
基于 Few-Shot 提示工程的电商用户评论情感分类器
实验目的：掌握 ChatPromptTemplate 与 FewShotChatMessagePromptTemplate 的组合使用方法，学会通过少样本示例控制大模型的输出格式与推理逻辑。
实验内容：
1、设计一个针对电商商品评论的提示词模板，包含至少 3 组少样本示例（包含评论文本、对应的情感标签如“好评/差评/中立”，以及简要的原因分析）。
2、使用 LangChain 调用大语言模型，实现对输入的任意用户评论进行准确分类并给出判定理由。
观察并比较“无示例（Zero-Shot）”与“少样本（Few-Shot）”在复杂语义识别下的准确率差异。
"""
import os
from langchain_core.prompts import(
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate
)
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
# 加载 .env 文件里面全部环境变量
load_dotenv("llm.env")

# 定义示例的格式化模板
examples = [
    {"input": "这个可以", "output": "好评，主要包含了可以，表示肯定"},
    {"input": "这个一般般", "output": "中立，主要包含了一般般，表示中立"},
    {"input": "这个不行", "output": "差评，主要包含了不行，表示否定"},
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
    ("system", "你是一个电商用户评论情感分类器，请根据下面的示例格式进行情感分类,并给出理由。"),
    few_shot_prompt,
    ("human", "{input}"),  # 用户的实际输入变量
])
# 调用Deepseek模型
model = ChatOpenAI(
    model="deepseek-v4-flash",
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    temperature=1  # 数学题通常设低温度，保证准确性
)
chain = final_prompt | model

text = "整体用下来中规中矩，产品功能基本都能满足预期。做工不算特别精致，但也没有明显瑕疵。发货速度还行，使用体验平平，没有特别惊喜，也没有踩雷，日常使用够用，性价比一般，有需要可以考虑入手。"
response = chain.invoke({"input": f"{text}"})
print(f"用户输入:{text}")
print(f"模型回答:{response.content}")

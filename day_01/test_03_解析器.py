# 输出解析器将大模型的原始输出转换为下游应用易于使用的格式
# 将LLM的文本输出转换为结构化信息(例如JSON、XML等)
# 将ChatMessage转换为纯字符串;
# 将信息外的内容(如自定义函数调用中返回的额外信息)转换为字符串
# 一个将以返号分隔的字符串转换为列表的解析器。
# 链模块：chat_prompt | llm |解析器
# 导入操作系统模块，用来设置环境变量（存放API密钥）
import os
# 导入输出解析器基类，我们要继承它实现自定义解析器
from langchain_core.output_parsers import BaseOutputParser
# 导入ChatOpenAI，DeepSeek兼容OpenAI接口格式，所以用这个类来调用
from langchain_openai import ChatOpenAI
# 导入HumanMessage，代表用户发送的消息对象
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
# 加载 .env 文件里面全部环境变量
load_dotenv("llm.env")
# 初始化DeepSeek配置
llm = ChatOpenAI(
    model="deepseek-v4-flash",
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    temperature=0.9,
)


# 自定义解析器
# 继承BaseOutputParser，实现自己的解析逻辑
# BaseOutputParser是LangChain所有输出解析器的父类
class CommaSeparatedListOutputParser(BaseOutputParser):
    """将LLM输出的内容解析为列表"""
    def parse(self, text: str):
        """解析LLM调用的输出"""
        # strip()去除首尾空格，split(",")按逗号分割
        return text.strip().split(",")


if __name__ == "__main__":
    # 定义用户提问文本
    text = "给生产杯子的公司取三个合适的中文名字，以逗号分隔的形式输出。"
    # 封装成HumanMessage消息对象，Chat模型需要消息对象作为输入
    messages = [HumanMessage(content=text)]
    # 调用大模型，发送消息，获取模型返回响应对象
    llms_response = llm.invoke(messages)
    # 打印大模型原始返回内容，调试用，方便看AI到底输出了什么
    print("大模型原始返回内容：", llms_response.content)
    # 创建自定义解析器实例，调用parse方法解析模型返回的content文本
    result = CommaSeparatedListOutputParser().parse(llms_response.content)
    # 打印解析完成之后的Python列表结果
    print("解析之后的列表：", result)

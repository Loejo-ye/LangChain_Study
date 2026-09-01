from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableLambda
from config import llm


def generate_summary(text):
    prompt = f"""
    请对下面的文章进行总结
    要求:1、提取文章核心内容
    2、控制在100字以内
    3、使用中文回答
    文章:{text}
    """
    # 调用大模型，传入字符串提示词，得到模型返回消息对象
    response = llm.invoke(prompt)
    return response.content


# 定义关键词提取函数
def generate_keywords(text):
    prompt = f"""
    请从下面的文章中提取5个关键词。
    只输出关键词，不要解释
    文章:{text}
    """
    response = llm.invoke(prompt)
    return response.content


summary_chain = RunnableLambda(generate_summary)
keywords_chain = RunnableLambda(generate_keywords)


# 自定义分析主流程函数u(手动串联两条任务)
def text_analysis_chain(text):
    summary = summary_chain.invoke(text)
    keywords = keywords_chain.invoke(text)
    return {
        'summary': summary,
        'keywords': keywords
    }


if __name__ == '__main__':
    article = """
    人工智能正在快速改变软件开发行业。
    大语言模型可以辅助程序员完成代码生成、软件测试、技术文档编写以及程序调试。
    随着大语言模型技术的发展，越来越多的软件开发企业开始使用AI辅助工具提高开发效率
    """
    print("=" * 70)
    print("自定义Chain")
    print("=" * 70)

    result = text_analysis_chain(article)
    print("\n 【文章摘要】")
    print(result["summary"])
    print("\n 【关键词】")
    print(result["keywords"])








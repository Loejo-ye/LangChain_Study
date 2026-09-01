"""
基于 LCEL 的多模式文本生成与流式响应系统
实验目的：掌握 LangChain 表达式语言（LCEL）的基础构件，熟悉 ChatPromptTemplate 与 StrOutputParser 的链式组合，并掌握 invoke、stream、batch 及 astream 等执行机制的应用场景。
实验任务：
1、链构建与 Schema 检验：设计一个“多语言科技新闻摘要助手”的提示词模板，结合大模型与字符串输出解析器构建基础 LCEL 链，并输出查看该链的输入 Schema 结构。
2、单次与批量调用：使用 invoke 完成单条新闻的摘要提取；构造包含 4 篇不同领域文本的输入列表，使用 batch 实现高效批量并发处理。
3、同步与异步流式输出：分别使用 stream 与 astream（结合 asyncio） 实现控制台实时打字机式的文本渲染。
"""
import asyncio
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config import llm

prompt = ChatPromptTemplate.from_template(
    """
    你是一个多语言科技新闻摘要助手.
    根据用户传入的新闻文本：{topic}
    进行摘要提取；构造包含 4 篇不同领域文本的输入列表
    """

)

chain = prompt | llm | StrOutputParser()

# print("输入Schema")
# print(chain.input_schema.schema())

print("=" * 60)
print("1.invoke")
print("=" * 60)
result = chain.invoke({
    "topic": "2026‑09‑019 月 1 日，2026 年全国科普月正式开启，主题为 “科技改变生活 创新赢得未来”，活动为期一个月。本次活动由中国科协联合 34 家单位共同举办，主场设置科学家实物展、脑科学科普、航天实践等多项专题活动。全国各地将开展校园科普、场馆巡展、线上科普直播等项目，面向全民普及科学知识，提升公众科学素养。"
})
print(result)

print("\n" + "=" * 60)
print("2.batch")
print("=" * 60)
topics = [
    {"topic": "2026‑09‑01，2026 全国科普月正式启动，主题为 “科技改变生活 创新赢得未来”，活动为期一个月，本次活动由中国科协联合 34 家单位共同举办，主场设置科学家实物展、脑科学科普、航天实践等多项专题活动，全国各地将开展校园科普、场馆巡展、线上科普直播等项目，面向全民普及科学知识，提升公众科学素养。"},
    {"topic": "2026‑09‑01，2026 国家网络安全宣传周 9 月 14 日开幕，将于 9 月 14 日‑20 日在全国开展，主会场设在山东济南，今年主题聚焦智能时代网络防护，设置 1 场高峰论坛、15 场专题分论坛，活动将围绕人工智能安全、个人信息保护、反诈防护等内容开展科普宣传，面向企业、校园、社区普及网络安全知识，发布多项网络安全技术成果。"},
    {"topic": "2026‑08‑31，商务部等七部门联合发布实施意见，提出 20 条举措扩大国内商品消费，文件提出培育绿色、智能、健康十万亿级消费市场，做大汽车、家电、服装等万亿级消费品类，通过完善城乡消费场所、健全消费品标准、强化财政金融支持，激发居民消费潜力，推动国内消费市场持续增长。"},
    {"topic": "2026‑09‑01，2026 中国农民丰收节金秋消费季即将开启，将于本周启动，全国主会场落地上海，活动从 9 月持续至年底，包含农产品市集、乡村骑行、郊野文旅体验等内容，一方面把全国各地特色农产品引入城市展销，另一方面鼓励市民走进乡村，打通 “农产品进城、市民下乡” 双向通道，展现乡村振兴成果。"},
]

results = chain.batch(topics)
for i, result in enumerate(results):
    print(f"\n文章{i+1}:")
    print(result)

print("\n" + "=" * 60)
print("3.Stream")
print("=" * 60)
# 遍历流式返回的数据块chunk
for chunk in chain.stream({"topic": "2026‑09‑019 月 1 日，2026 年全国科普月正式开启，主题为 “科技改变生活 创新赢得未来”，活动为期一个月。本次活动由中国科协联合 34 家单位共同举办，主场设置科学家实物展、脑科学科普、航天实践等多项专题活动。全国各地将开展校园科普、场馆巡展、线上科普直播等项目，面向全民普及科学知识，提升公众科学素养。"}):
    print(chunk, end="", flush=True)
print()


# 4.stream:异步流式输出
async def async_stream():
    print("\n" + "=" * 60)
    print("4.astream")
    print("=" * 60)

    async for chunk in chain.astream({"topic": "AI Agent"}):
        print(chunk, end="", flush=True)

    print()
# asyncio.run()启动异步时间循环，执行异步函数
asyncio.run(async_stream())


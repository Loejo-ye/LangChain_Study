# 导入langchain核心的提示词模板类
# PromptTemplate：用于普通字符串提示模板
# ChatPromptTemplate：用于聊天对话格式（system、human、ai消息）
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate

print("测试提示词模板")

# ========== 1. PromptTemplate 普通字符串模板 ==========
# from_template()：静态方法，传入模板字符串，创建PromptTemplate对象
# {produce} 是占位变量，后续调用format传入对应值进行替换
prompt = PromptTemplate.from_template("给生产{produce}的公司取一个名字。")

# format()：传入变量字典，把模板里的占位符替换成真实内容，返回普通字符串
result_str = prompt.format(produce="杯子")

# 打印替换完成后的完整提示字符串
print(f"生成的字符串：{result_str}")


# ========== 2. ChatPromptTemplate 聊天消息模板 ==========
# 聊天场景会区分角色：system(系统提示)、human(用户提问)、ai(AI回复)
# 定义系统角色模板，两个占位变量：输入语言、输出语言
system_template = "你是一个能将{input_language}翻译成{output_language}的助手"
# 用户消息模板，占位变量text代表用户输入的文本
human_template = "{text}"

# from_messages()：传入消息列表，构建聊天提示模板
# 列表每一项格式：(角色类型, 模板字符串)
chat_prompy = ChatPromptTemplate.from_messages([
    ("system", system_template),  # system角色：设定AI身份
    ("human", human_template),    # human角色：用户输入内容
])

# format_messages()：传入所有占位变量，返回**消息对象列表**（不是普通字符串！）
# 每个消息对象包含：角色、消息内容，适配大模型Chat接口
messages = chat_prompy.format_messages(
    input_language="中文",
    output_language="英文",
    text="我爱编程"
)

# 循环遍历打印每一条消息对象
# 打印出来会看到：SystemMessage(content=xxx)、HumanMessage(content=xxx)
for msg in messages:
    print(msg)




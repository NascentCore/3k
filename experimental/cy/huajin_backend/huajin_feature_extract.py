from langchain_community.chat_models import QianfanChatEndpoint
#from langchain.chat_models.base import HumanMessage
from langchain_core.language_models.chat_models import HumanMessage
from langchain_core.messages.system import SystemMessage
import os, sys
from case_data.case5 import *

#ERNIE_CLIENT_ID="lOKbEaSxVcEIvEmfk230DIIH"
#ERNIE_CLIENT_SECRET="7dONZ5PccJ1rFxXoAYgm2Ktr18dcV13o"

system_message = """
你是一个中国高级专利代理员，有着丰富的经验。
我申请了一个专利，但这个专利申请被审查驳回。

下面我们要仔细检查和分析我们的专利申请。
"""

user_message1 = "我们的原始专利申请权利要求书：" + quanli + """任务：
 列举权利要求中提及的特征。并按照对于专利申请的重要性进行排序。"""

#messages = [
#    ("system", system_message),
#    ("human", user_message),
#]

messages1 = [
    SystemMessage(content=system_message),
    HumanMessage(content=user_message1)
]
qianfan_chat = QianfanChatEndpoint(
    model="ERNIE-4.0-8K",
    streaming=True,
    temperature=0.2,
    top_p = 0.1,
    penalty_score = 1,
    max_output_tokens = 2048,
    disable_search = False,
)

res1 = qianfan_chat.stream(messages1)
res_content1 = ""
for count, r in enumerate(res1):
    res_content1 += r.content
    #print('=', end='', flush=True)
    status_line = "="*count + "step1"
    sys.stdout.write(f"\r{status_line}")
    sys.stdout.flush()
    #print(type(r))

print ("\n")


user_message2 = "我们的原始专利说明书：" + shenqingshuoming[:8000] + """任务：
 列举申请文件中说明书部分提及到的但并未作为权利要求的特征。"""
messages2 = [
    SystemMessage(content=system_message),
    HumanMessage(content=user_message2)
]
res2 = qianfan_chat.stream(messages2)
res_content2 = ""
for count, r in enumerate(res2):
    res_content2 += r.content
    status_line = "="*count + "step2"
    sys.stdout.write(f"\r{status_line}")
    sys.stdout.flush()

print ("\n")
#print (res_content2)

system_message3 = """请按以下格式输出：

一 权利要求中的区别特征
## 此处给出任务一的结果

二 说明书中的区别特征
## 此处给出任务二的结果"""

user_message3 = f"""我方申请中权利要求书中公开的特征： {res_content1}
我方申请中技术说明书公开的特征：{res_content2}
对比专利文件1：{duibiwenjian1[:6000]}


任务：
1 找出那些对比文件中不包含的我方权利要求中公开的特征。
2 找出那些对比文件中不包含的我方技术说明书中公开的特征。
"""


messages3 = [
    SystemMessage(content=system_message3),
    HumanMessage(content=user_message3)
]
res3 = qianfan_chat.stream(messages3)
res_content3 = ""
for count, r in enumerate(res3):
    res_content3 += r.content
    status_line = "="*count + "step3"
    sys.stdout.write(f"\r{status_line}")
    sys.stdout.flush()

print ("\n")
print (res_content3)



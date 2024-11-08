from llm import call_llm
from config import SERVICE_LIST 

def domain_check(ipt) :
    domains = ",".join(SERVICE_LIST)
    s = f"假设你是一个机器人，以下你的服务的范围【 {domains} 】, 请判断此问题 【 {ipt} 】是否在你的服务范围内，如果是请说YES，如果不是请说NO。"
    res = call_llm(s)
    return res == "YES"


if __name__ == "__main__" :
    print(domain_check("明天适合洗车吗？"))
    print(domain_check("明天天气怎么样？"))



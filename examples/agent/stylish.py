from llm import call_llm 
from config import STYLISH_DESC


def stylish_output(ipt) :
    return call_llm(ipt= f"【{STYLISH_DESC}】请按照此风格，重新描述以下的信息：【{ipt}】")


if __name__ == "__main__" :
    print(stylish_output("明天天气睛转多云，适合洗车。"))
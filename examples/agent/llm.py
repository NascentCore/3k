import requests
from config import LLM_URL


def call_llm(ipt):
    print("llm ipt : " , ipt)
    # 请求头
    headers = {
        'Content-Type': 'application/json',
    }

    # 请求体数据
    data = {
        'message': ipt,
        'rag': False
    }

    # 发送POST请求
    response = requests.post(LLM_URL, headers=headers, json=data)

    # 检查响应是否成功
    if response.status_code == 200:
        response_data = response.json()

        # 提取choices字段中第一个元素的message字段下的content字段
        content = response_data['choices'][0]['message']['content']
        print("llm output : " , content)
        return content
    else:
        return "请求失败，状态码：{}".format(response.status_code)


if __name__ == "__main__" :
    # 调用函数并打印结果
    content = call_llm("tell me a story")
    print(content)






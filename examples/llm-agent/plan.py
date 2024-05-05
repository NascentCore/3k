from datetime import datetime
import requests
from config import LLM_URL


def planning(user_query, tool_list):
    today = datetime.now().strftime("今天是%Y-%m-%d，%A")

    query = f'''
    用户原始问题: {user_query}
    背景信息: {today}
    工具列表: {tool_list}
    你是一个任务编排的专家，你要根据用户原始问题和工具列表，来生成任务调用的过程，<answer>代表上一个工具的输出，这是一个你收到的输入示例: 
        {{用户原始问题: 下个月1号适合洗车吗？;背景信息: 今天是2024-01-31 星期三}}，
    你最终的输出是一个json数组，不要有其他的多余回答: 
    [{{"weather_assistant":{{"input":"北京,2024-02-01"}}}},{{"common_sense_assistant":{{"input":"<answer> 适合洗车吗？"}}}}];
    '''

    print("query:", query)

    # Prepare data for sending the request to OpenChat
    payload = {
        "message": query,
        "rag": False
    }

    tries = 5
    for i in range(tries) :
        res = None
        try :
            # Send request to OpenChat
            response = requests.post(LLM_URL, json=payload)

            plan = response.json()
            content = plan['choices'][0]['message']['content']
            res = eval(content)
        except Exception as e:
            print("exception occured:", e)
        else:
            return res
    print(f"plan failed after {tries} tries")
    return None

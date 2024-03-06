import os
import requests
from city import city_dict
from config import LLM_URL


def query_weather(input_str):
    """
    查询指定日期和地点的天气
    
    :param input_str: 2024-02-25,杭州
    :return: 多云
    """
    try:
        date, location = [item.strip() for item in input_str.split(',')]
    except ValueError:
        return "输入格式错误，正确格式为：'YYYY-MM-DD, 地点'"
    
    city_code = 110000
    for name, code in city_dict.items():
        if name.startswith(location):
            city_code = code
            break

    key = os.getenv('WEATHER_KEY')
    api_url = "https://restapi.amap.com/v3/weather/weatherInfo"
    
    params = {
        "key": key,
        "city": city_code,
        "extensions": "all"
    }
    
    try:
        response = requests.get(api_url, params=params)
        # 检查响应状态码
        response.raise_for_status()
        
        # 解析JSON格式的响应
        weather_data = response.json()
        
        for item in weather_data["forecasts"][0]["casts"]:
            if item["date"] == date:
                return item["dayweather"]

        return "只能查询未来三天的天气预报"
    except requests.RequestException as e:
        print(f"请求天气API时出错: {e}")
        return None

def call_inference(question):
    """
    调用推理API并返回结果。
    
    :param question: 明天是晴天，适合洗车吗?
    :return: 是的，明天是晴天的话，非常适合洗车
    """
    headers = {"Content-Type": "application/json"}
    data = {
        "message": question,
        "rag": False
    }
    
    try:
        response = requests.post(LLM_URL, headers=headers, json=data)
        response.raise_for_status()

        # 解析JSON格式的响应
        weather_data = response.json()
        return weather_data["choices"][0]["message"]["content"]
    except requests.RequestException as e:
        print(f"请求推理API时发生错误: {e}")
        return None

if __name__ == '__main__':
    print(query_weather("2024-03-06,杭州"))
    print(call_inference("明天是晴天，适合洗车吗?"))

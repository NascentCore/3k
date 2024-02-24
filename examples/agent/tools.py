import os
import requests
from city import city_dict

def query_weather(date, location):
    """
    查询指定日期和地点的天气
    
    :param date: 指定的日期，格式为 'YYYY-MM-DD'
    :param location: 查询天气的地点
    :return: 天气文字描述
    """
    city_code = 110000
    for name, code in city_dict.items():
        if name.startswith(location):
            city_code = code
            break

    print(city_code)
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

if __name__ == '__main__':
    result = query_weather("2024-02-25", "杭州")
    print(result)

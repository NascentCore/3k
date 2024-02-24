import unittest
from unittest.mock import patch, MagicMock
from tools import query_weather

class TestWeatherQuery(unittest.TestCase):

    @patch('tools.os.getenv')
    @patch('tools.requests.get')
    def test_query_weather(self, mock_get, mock_getenv):
        # 设置模拟环境变量的返回值
        mock_getenv.return_value = 'test_weather_key'

        # 设置模拟的响应数据
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "forecasts": [{
                "casts": [
                    {"date": "2024-02-25", "dayweather": "晴转多云"},
                    {"date": "2024-02-26", "dayweather": "小雨"},
                    # ...
                ]
            }]
        }
        mock_get.return_value = mock_response

        # 测试晴转多云的情况
        weather = query_weather("2024-02-25", "杭州")
        self.assertEqual(weather, "晴转多云")

        # 测试无法查询的日期
        weather = query_weather("2024-02-29", "杭州")
        self.assertEqual(weather, "只能查询未来三天的天气预报")

if __name__ == '__main__':
    unittest.main()

from plan import planning

tool_list = [
    {
        "name": "weather_assistant",
        "describe": "I can query the weather by city and date",
        "input_example": "北京,2024-02-22",
        "output_example": "晴天"
    },
    {
        "name": "common_sense_assistant",
        "describe": "I have a lot of common sense knowledge about daily life",
        "input_example": "明天是晴天，适合洗车吗?",
        "output_example": "是的，明天是晴天的话，非常适合洗车"
    }
]

print(planning("下周二适合洗车吗？", tool_list))
from tools import query_weather

# 定义函数注册器
class FunctionRegistry:
    def __init__(self):
        self._registry = {}

    def register(self, func):
        """
        注册工具函数到注册器中
        :param func: 要注册的函数
        """
        if func.__name__ in self._registry:
            raise KeyError(f"Function '{func.__name__}' is already registered.")
        self._registry[func.__name__] = func

    def call(self, func_name, *args, **kwargs):
        """
        通过函数名称调用注册的函数。
        :param func_name: 已注册的函数名称
        :param args: 传递给函数的位置参数
        :param kwargs: 传递给函数的关键字参数
        :return: 被调用函数的返回值
        """
        func = self._registry.get(func_name)
        if func is None:
            raise KeyError(f"Function '{func_name}' is not registered.")
        return func(*args, **kwargs)

# 创建注册器实例
registry = FunctionRegistry()

# 注册query_weather函数
registry.register(query_weather)

# 通过registry.call来调用query_weather函数
# 例如，查询2024年2月25日杭州的天气
# weather = registry.call('query_weather', "2024-02-25", "杭州")
# print(weather)

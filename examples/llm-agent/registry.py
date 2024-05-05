from tools import query_weather, call_inference

# 定义函数注册器
class FunctionRegistry:
    def __init__(self):
        self._registry = {}
        self.tool_list = []

    def register(self, func):
        """
        注册工具函数到注册器中，并根据函数的文档字符串更新工具列表。
        :param func: 要注册的函数
        """
        if func.__name__ in self._registry:
            raise KeyError(f"Function '{func.__name__}' is already registered.")
        self._registry[func.__name__] = func

        # 提取工具信息并添加到工具列表
        tool_info = self.extract_tool_info(func)
        self.tool_list.append(tool_info)

    def extract_tool_info(self, func):
        """
        根据函数的文档字符串提取工具信息。
        :param func: 已注册的函数
        :return: 包含工具信息的字典
        """
        # 提取文档字符串中的描述、输入示例和输出示例
        doc_lines = func.__doc__.split('\n')
        print(doc_lines)
        name = func.__name__
        describe = doc_lines[1].strip()
        input_example = doc_lines[3].split(": ")[1].strip('"')
        output_example = doc_lines[4].split(": ")[1].strip('"')
        
        return {
            "name": name,
            "describe": describe,
            "input_example": input_example,
            "output_example": output_example
        }

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

# 注册工具函数
registry.register(query_weather)
registry.register(call_inference)

# 获取tool_list
# print(registry.tool_list)
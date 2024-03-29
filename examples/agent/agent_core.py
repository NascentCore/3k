from domain import domain_check
from stylish import stylish_output
from config import TOOL_LIST , SERVICE_LIST
from plan import planning
from registry import registry

def get_relative_memory(ipt) :
    return "" 

def process_req(req) :
    # check the req 
    # check_res = domain_check(req)
    # if not check_res :
    #     return "Your request is beyond the scope of our service" , False
    
    # get context from memory
    context = get_relative_memory(req)
    enriched_query = f"请在如下的信息背景【 {context} 】下， 解决以下问题【 {req} 】"

    # split into tasks 
    tasks = planning(enriched_query , registry.tool_list)
    print("tasks : " , tasks)
    if type(tasks) == str :
        tasks = eval(tasks)
    # do the tasks
    task_output = enriched_query
    for task in tasks :
        print(task)
        for tool_id in task :
            tool_ipt = task[tool_id]
            #如果没给输入，则用上一次的输出作为输入
            tool_ipt = tool_ipt.get("input" , "")
            if tool_ipt == "" :
                tool_ipt = task_output
            #将其中的 <answer> ， 替换成上一次的输出 {\"input\":\"<answer> 是的，晴天的话，非常适合洗车\"}
            tool_ipt = tool_ipt.replace("answer" , task_output)
            task_output = registry.call(tool_id , tool_ipt)
            
    # stylish the result 
    res = stylish_output(task_output)

    return res


def get_service_list() : 
    return SERVICE_LIST

def register_tool() :
    pass 


if __name__ == "__main__" :
    # 调用函数并打印结果
    tasks = [{"query_weather":{"input":"2024-02-26,杭州"}},{"call_inference":{"input":"明天是晴天，适合洗车吗？"}}]
    task_output = ""
    for task in tasks :
        print(task)
        for tool_id in task :
            tool_ipt = task[tool_id]
            #如果没给输入，则用上一次的输出作为输入
            tool_ipt = tool_ipt.get("input" , "")
            #将其中的 <answer> ， 替换成上一次的输出 {\"input\":\"<answer> 是的，晴天的话，非常适合洗车\"}
            tool_ipt = tool_ipt.replace("answer" , task_output)
            print(tool_id , tool_ipt)




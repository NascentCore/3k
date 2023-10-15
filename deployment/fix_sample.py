import yaml
import os
import sys
#in.yaml文件与当前的文件在同一个目录下
 
# 获取当前脚本所在文件夹路径
# print (sys.argv[1])
curPath = os.path.realpath(sys.argv[1])
# 获取yaml文件路径
yamlPath = os.path.join(curPath, "sample.yaml")
 
'''
读取数据
'''
def get_dict(node, usr, pwd, ip):
    d = {}   #in.yaml文件里面存了字典
    f = open(yamlPath, 'r', encoding='utf-8')
    d = yaml.load(f.read(), Loader=yaml.FullLoader)
    f.close()

    data = {'name': node, 'address': ip, 'internalAddress': ip, 'user': usr, 'password': pwd}
    d['spec']['hosts'].append(data)
    d['spec']['roleGroups']['worker'].append(node)

    with open(yamlPath, 'w') as f:
        yaml.dump(d, f)    #将Python中的字典或者列表转化为yaml格式的数据
        f.close()

    print("add : " ,data)


if __name__ == "__main__":
    get_dict(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
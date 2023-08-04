import os
import sys
import yaml

# execute command, and return the output  
def execCmd(cmd):  
    r = os.popen(cmd)  
    text = r.read()  
    r.close()  
    return text  

def addSC(path):
    cmd = 'grep "kind: PersistentVolumeClaim" ' + path + ' -r'
    print(cmd)
    out  = execCmd(cmd).split('\n')
    filenames = [ line.split(":")[0] for line in out ]
    
    print('需要修改的文件名：')

    for name in filenames:
        if name == '' :
            continue
        print(name)
        d = {}   #in.yaml文件里面存了字典
        f = open(name, 'r', encoding='utf-8')
        d = yaml.load(f.read(), Loader=yaml.FullLoader)
        f.close()

        d['spec']['storageClassName'] = 'local-path'

        with open(name, 'w') as f:
            yaml.dump(d, f)    #将Python中的字典或者列表转化为yaml格式的数据
            f.close()


if __name__ == '__main__':
    addSC(sys.argv[1])
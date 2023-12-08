# Ansible

Ansible files to setup development environment on Linux and Mac.

## user.yaml
### 功能
- 在 managed_nodes 机器上创建用户
- 为该用户授予sudo权限
- 将用户加入 docker 用户组，如果该用户组存在
- 为该用户创建ssh密钥对
- 将公钥复制到所有节点上，使节点间ssh互信

### 用法
1. 创建你的 inventory 文件，示例如下：
```bash
$ cat my-hosts
[managed_nodes]
worker1
worker2
```

2. 执行如下命令
```bash
USER_NAME=my-user-name USER_PASSWORD=my-password ansible-playbook -i my-hosts user.yml
```

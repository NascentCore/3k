# e2e 测试

## 编译成一个 test 二进制程序

```bash
go test e2e/ -c -o e2e
```

## 使用

```bash
test -test.v --namespace="" --enable-ib=true

```

## 镜像

需要维护测试用例的镜像，并提供导出列表。

## case 用例

1. ib 节点连通性
2. pytorchjob
3. mpijob
4. 常见模型
   - gpt3
     - 单机单卡
     - 单个多卡
     - 多机多卡
   - bert
     - 单机多卡
     - 单机多卡

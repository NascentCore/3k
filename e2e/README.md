# e2e 测试

e2e 测试模块旨在运行 3k 平台的测试 case,基于 e2e-framework 项目。

## 编译成一个 test 二进制程序

```bash
go test e2e/ -c -o e2e-test
```

## 使用

```bash
# 打印帮助， go test的参数通过`-test.`的形式指定
e2e-test --help 2&>1 | less

# 创建3k-e2e，运行所有case
e2e-test -test.v --namespace="3k-e2e" --enable-ib=true

```

## 镜像

# TODO: @sxwl-donggang

需要维护测试用例的镜像，并提供导出列表。并且镜像仓库地址可以参数配置

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

### 编写一个 case

## 问题

1. 无法指定测试 case 的顺序，但是在 e2e 测试场景，指定测试 case 顺序的需求十分普通遍，这是因为 e2e-framework 基于 golang test，而 golang test 的设计思想是面向单元测试，且`测试文件之间的顺序不应该有所依赖`，所以无法指定一个包下测试文件。package 下的所有 test 文件默认执行顺序按照[文件名开头字母排序](https://stackoverflow.com/questions/29268881/how-to-enforce-testing-order-for-different-tests-in-a-go-framework)。

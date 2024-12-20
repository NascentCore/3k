# litellm-stack部署演示

- 下载模型到本地 `/mnt/models` 目录
```shell
cd /mnt/models
git lfs install
git clone https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3 mistral-7b-instruct
git clone https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct meta-llama-3.1-8b-instruct
```

- 部署 litellm-stack
```shell
# 添加 helm 仓库
helm repo add sxwl https://sxwl-ai.oss-cn-beijing.aliyuncs.com/charts/
helm repo update

# 查看并保存内容到 values.ymal
helm show values sxwl/litellm-stack | tee values.yaml

# 如果模型下载路径有变更，请同步修改 values.yaml 中 models.model_path 的值

# 部署 litellm-stack
helm install litellm-stack sxwl/litellm-stack -f values.yaml

# 查看 pods 状态
kubectl get pod -n litellm-stack

# 查看 service
kubectl get svc -n litellm-stack
```

- 现在可通过 http://{node_ip}:31000 访问 litellm 管理后台
  - 默认用户名：admin
  - 默认密码：sk-1234（可通过 values.yaml 中 litellm.master_key 修改）

- 运行测试脚本
```shell
# 安装 openai 库
pip install openai

# 修改 chat_test.py 中 base_url 中的 node_ip 为实际节点 IP
# 运行测试
$ python chat_test.py
Model: llama-31-8b-instruct
Question:What is the capital of France?
Response: The capital of France is Paris.

Model: mistral-7b-instruct
Question:Explain quantum computing in simple terms.
Response: **What is Quantum Computing?**

Imagine you have a huge library with an infinite number of books. Each book represents a possible solution to a problem. In a classical computer, you would have to go through each book one by one, searching for the correct solution. This process can take a long time, especially if the library is huge.

**Quantum Computing: The Magic of Superposition**

A quantum computer is like a super-powered librarian. It can look at all the books in the library simultaneously
```

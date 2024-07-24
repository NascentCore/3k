# API Test

This is a test for API.

## Usage
1. 下载 3k 仓库
```bash
git clone https://github.com/NascentCore/3k.git
cd 3k
```

2. 安装 python 依赖
```bash
pip install requests
```

3. 配置环境变量

API_URL 和 TOKEN 根据测试的环境填写相应的值

```bash
export SXCLOUD_API_URL='<API_URL>'
export SXCLOUD_API_TOKEN='<TOKEN>'
export FEISHU_WEBHOOK='<WEBHOOK>'
```

4. 执行测试脚本
```bash
python api_test.py
```


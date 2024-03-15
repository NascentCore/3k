import requests
import time
import numpy as np
import jieba
from torch.utils.tensorboard import SummaryWriter

# 服务的URL
url = "http://10.233.57.192/v1/chat/completions"

# 准备测试数据
input_data = {
    "model": "openchat_3.5",
    "messages": [{"role": "user", "content": "明天是晴天，适合洗车吗?"}]
  }

measurements = []

writer = SummaryWriter('/tmp/logs/p-eval')

for _ in range(10):  # 进行10次测量
    start_time = time.time()
    response = requests.post(url, json=input_data)
    first_token_time = time.time() - start_time
    measurements.append(first_token_time)

response_tokens = list(jieba.cut(response.json()['choices'][0]['message']['content'], cut_all=False))
num_tokens = len(response_tokens)

# 计算TTFT和TPOT
ttft = sum(measurements) / len(measurements)
tpot = ttft / num_tokens if num_tokens > 0 else 0

print(f"Time To First Token (TTFT): {ttft:.4f} seconds")
print(f"Time Per Output Token (TPOT): {tpot:.4f} seconds/token")

writer.add_histogram('Performance/TTFT (seconds)', ttft)
writer.add_histogram('Performance/TPOT (seconds/token)', tpot)

# 对Latency和Throughput的评测
num_requests = 100
latencies = []

start_time = time.time()
for _ in range(num_requests):
    request_start = time.time()
    response = requests.post(url, json=input_data)
    request_end = time.time()
    latencies.append(request_end - request_start)

end_time = time.time()

# 计算Latency和Throughput
average_latency = np.mean(latencies)
throughput = num_requests / (end_time - start_time)

print(f"Average Latency: {average_latency:.4f} seconds")
print(f"Throughput: {throughput:.2f} requests per second")

writer.add_histogram('Performance/Latency (seconds)', average_latency)
writer.add_histogram('Performance/Throughput (requests/sec)', throughput)
writer.close()

import os
import numpy as np
import pandas as pd
import argparse
import requests

from mp_utils import choices, format_example, gen_prompt, softmax, run_eval

def eval_openchat_api(model_name, subject, dev_df, test_df, num_few_shot, max_length, cot, api_url):
    cors = []
    all_preds = []

    for i in range(test_df.shape[0]):
        prompt_end = format_example(test_df, i, subject, include_answer=False, cot=cot)
        prompt = gen_prompt(dev_df=dev_df,
                            subject=subject,
                            prompt_end=prompt_end,
                            num_few_shot=num_few_shot,
                            max_length=max_length,
                            cot=cot)
        label = test_df.iloc[i, test_df.shape[1] - 1]

        # 准备请求数据
        data = {"model": model_name, "messages": [{"role": "user", "content": prompt}]}
        print(prompt)

        # 发送请求到推理服务
        response = requests.post(api_url, json=data)
        if response.status_code == 200:
            # 解析响应数据
            #pred = response.json().get('prediction', '')
            pred = response.json()["choices"][0]["message"]["content"].split(".")[0]
            print(f"{pred}, 参考答案：{label}")
            if pred and pred in choices:
                cors.append(pred == label)
            all_preds.append(pred.replace("\n", ""))
        else:
            print(f"Error calling inference API: {response.text}")

    acc = np.mean(cors)
    print(f"Average accuracy: {acc:.3f} - {subject}")
    return acc, all_preds, None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="openchat_3.5")
    parser.add_argument("--api_url", type=str, default="http://10.233.57.192/v1/chat/completions")
    parser.add_argument("--data_dir", type=str, default="../data")
    parser.add_argument("--save_dir", type=str, default="../results/OpenChatAPI")
    parser.add_argument("--num_few_shot", type=int, default=0)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--cot", action='store_true')
    args = parser.parse_args()

    # 适配到推理服务API调用
    run_eval(None, None, eval_openchat_api, args)


import json
import os
import argparse
from openai import OpenAI
from loguru import logger

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def get_model_responses(input_file, output_file, model_name):
    # 初始化OpenAI客户端
    client = OpenAI(
        api_key="",
        base_url="https://ark.cn-beijing.volces.com/api/v3/"
    )

    # 读取输入数据
    data = read_jsonl(input_file)

    # 存储结果的列表
    results = []

    for count, item in enumerate(data):
        if (count / 100 == 0):
            print ("infer %d" % count)

        try:
            # 获取用户问题
            user_content = None
            for message in item['messages']:
                if message['role'] == 'user':
                    user_content = message['content']
                    break

            if not user_content:
                continue

            # 调用API获取回复
            response = client.chat.completions.create(
                    #{"role": "system", "content": "你是豆包，是由字节跳动开发的 AI 人工智能助手"},
                messages=[
                    {"role": "user", "content": user_content},
                ],
                model=model_name,
            )

            # 获取模型回复
            model_response = response.choices[0].message.content

            # 获取原始助手回复
            original_assistant_response = None
            for message in item['messages']:
                if message['role'] == 'assistant':
                    original_assistant_response = message['content']
                    break

            # 保存结果
            result = {
                'user_content': user_content,
                'model_response': model_response,
                'original_response': original_assistant_response
            }
            results.append(result)

            logger.info(f"Processed one item successfully")

        except Exception as e:
            logger.error(f"Error processing item: {str(e)}")
            continue

    # 保存结果到文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info(f"Results saved to {output_file}")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get model responses for input questions')
    parser.add_argument('--input_file', type=str, required=True, help='Path to input JSONL file')
    parser.add_argument('--output_file', type=str, required=True, help='Path to output JSON file')
    parser.add_argument('--model_name', type=str, required=True, help='Name of the model to use')

    args = parser.parse_args()

    get_model_responses(args.input_file, args.output_file, args.model_name)

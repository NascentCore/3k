import json

def process_json(input_file, output_file):
    # 读取输入文件
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 处理数据
    processed_data = []
    for item in data:
        processed_item = {
            'user_content': item['user_content'],
            'original_response': item['original_response']
        }
        processed_data.append(processed_item)
    
    # 写入输出文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    process_json('score_finetuned_2.json', 'evaluation.json') 
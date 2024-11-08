import json
import argparse

def process_dataset(input_path, output_path):
    # 读取原始json文件
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 为每条数据添加label字段
    for item in data:
        print("处理前:")
        print(item)
        item['label'] = 'health'
        print("处理后:")
        print(item)

    # 保存到新文件
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='/mnt/datasets/dataset.json', help='输入数据集路径')
    parser.add_argument('--output', type=str, default='/workspace/saved_model/new_dataset.json', help='输出数据集路径')
    args = parser.parse_args()

    process_dataset(args.input, args.output)

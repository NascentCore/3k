import json
import sys
import os
import evaluate
import jieba
from loguru import logger
#sys.path.append("../metric")
from metric.common_metric import bert_score


def calculate_rougeL_and_bert_scores(results_file, results_score_file):
    # 读取结果文件
    with open(results_file, 'r', encoding='utf-8') as f:
        results = json.load(f)

    # 准备两个列表存储预测和参考文本
    list_generated_text = []
    list_ground_truth_text = []

    # 收集所有文本对
    for item in results:
        list_generated_text.append(item['model_response'])
        list_ground_truth_text.append(item['original_response'])

    # 计算每个样本的rouge分数
    current_directory = os.getcwd()
    print("当前工作目录是:", current_directory)

    rouge = evaluate.load('./metric/huggingface/rouge')
    f = lambda text: list(jieba.cut(text))
    noaggregator_results = rouge.compute(predictions=list_generated_text, references=list_ground_truth_text, tokenizer=f, rouge_types=['rougeL'], use_aggregator=False)

    # 计算每个样本的bert分数并累加
    total_bert_score = 0
    total_rougeL_score = 0
    for i, item in enumerate(results):
        bert_score_value = bert_score(item['model_response'], item['original_response'])
        rougeL_score_value = noaggregator_results['rougeL'][i]
        total_bert_score += bert_score_value
        total_rougeL_score += rougeL_score_value
        item['bert'] = bert_score_value
        item['rougeL'] = rougeL_score_value
        print (bert_score_value)

    # 计算平均分数
    avg_bert_score = total_bert_score / len(results)
    avg_rouge_score = total_rougeL_score / len(results)

    # 打印平均分数
    print(f"Average RougeL score: {avg_rouge_score:.4f}")
    print(f"Average BERT score: {avg_bert_score:.4f}")

    # 将更新后的结果写回文件
    with open(results_score_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    #results_file = "./model_responses_tiny.json"
    #results_score_file = "./score_tiny.json"
    #results_file = "result_data/test_pretrained.json"
    #results_score_file = "result_data/score_pretrained_test.json"
    #results_file = "result_data/test_finetuned.json"
    #results_score_file = "result_data/score_finetuned.json"
    results_file = "test_finetuned_2.json"
    results_score_file = "score_finetuned_2.json"
    calculate_rougeL_and_bert_scores(results_file, results_score_file)
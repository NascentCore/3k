import os
import jieba
from text2vec import Similarity

def rouge_score(text1, text2):
    """
    计算两个文本之间的RougeL分数（简化版本）
    Args:
        text1: 第一个文本
        text2: 第二个文本
    Returns:
        float: RougeL分数
    """
    # 确保输入是字符串类型
    if isinstance(text1, dict):
        text1 = str(text1)
    if isinstance(text2, dict):
        text2 = str(text2)
    
    # 转换为字符串并确保是utf-8编码
    text1 = str(text1).encode('utf-8').decode('utf-8')
    text2 = str(text2).encode('utf-8').decode('utf-8')
    
    # 使用jieba分词
    words1 = list(jieba.cut(text1))
    words2 = list(jieba.cut(text2))
    
    # 找到最长公共子序列
    len1, len2 = len(words1), len(words2)
    dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]
    
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            if words1[i-1] == words2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    lcs_length = dp[len1][len2]
    
    # 计算F1分数
    if len1 == 0 or len2 == 0:
        return 0.0
        
    recall = lcs_length / len2  # 召回率
    precision = lcs_length / len1  # 精确率
    
    # 避免除零错误
    if recall + precision == 0:
        return 0.0
        
    # 计算F1分数
    f1 = 2 * recall * precision / (recall + precision)
    
    return f1

def bert_score(
    continuation: str,
    reference: str
) -> float:
    """
    Note:
        Requesting the network to connect to Hugging Face.
    """
    #sim = Similarity(model_name_or_path="src/text2vec-base-chinese")
    sim = Similarity(model_name_or_path=os.path.join(os.path.dirname(__file__), "sentence-transformers/bge-base-zh-v1.5"))
    score = sim.get_score(continuation, reference)
    return score
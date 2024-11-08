# encoding: UTF-8

from modelscope.pipelines import pipeline
word_segmentation = pipeline('word-segmentation',model='damo/nlp_structbert_word-segmentation_chinese-base')

input_str = '今天天气不错，适合出去游玩'
print(word_segmentation(input_str))

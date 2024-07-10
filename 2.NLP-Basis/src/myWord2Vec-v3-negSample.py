import numpy as np
import re
from collections import Counter
import random

# 示例文本数据
corpus = [
    "I love natural language processing",
    "Word2Vec is a technique for natural language processing",
    "Gensim is a library for topic modeling and document similarity",
    "Deep learning is useful for natural language understanding"
]

# 数据预处理：分词、去标点符号、小写化
def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    words = text.split()
    return words

corpus = [preprocess_text(doc) for doc in corpus]

# 构建词汇表
words = [word for doc in corpus for word in doc]
vocab = list(set(words))
vocab_size = len(vocab)

# 创建词与索引的映射
word_to_index = {word: i for i, word in enumerate(vocab)}
index_to_word = {i: word for i, word in enumerate(vocab)}

# 生成训练数据
def generate_training_data(corpus, window_size):
    training_data = []
    for sentence in corpus:
        sentence_length = len(sentence)
        for i, word in enumerate(sentence):
            context_words = []
            for j in range(max(0, i - window_size), min(sentence_length, i + window_size + 1)):
                if j != i:
                    context_words.append(sentence[j])
            for context_word in context_words:
                training_data.append((word, context_word))
    return training_data

window_size = 2
training_data = generate_training_data(corpus, window_size)

# 转换为索引
training_data = [(word_to_index[center], word_to_index[context]) for center, context in training_data]


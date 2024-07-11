
# Word2Vec - CBOW 模型: 周围词预测中心词
# 数学数据集：data/word2vec

import os
import jieba

import numpy as np
import pandas as pd
import os.path as osp

from tqdm import tqdm
from utils import softmax


def read_data(file):
    corpus = pd.read_csv(file,encoding="gbk",names=["data"])
    corpus = corpus["data"].tolist()

    res = []
    for text in corpus:
        word_cut = jieba.lcut(text)
        res.append(word_cut)

    return res

def word2index(corpus):
    word2idx = {"PAD":0, "UNK":1}
    for text in corpus:
        for word in text:
            word2idx[word] = word2idx.get(word, len(word2idx))

    idx2word = list(word2idx)

    return word2idx, idx2word


def word2onehot(num_words):
    return np.eye(num_words).reshape(num_words, 1, num_words)


# 获取词向量
def get_word_vector(word):
    word_index = word2idx[word]
    return W1[word_index]

if __name__ == "__main__":
    corpus = read_data(osp.join("data", "word2vec", "数学原始数据.csv"))
    
    word2idx, idx2word = word2index(corpus) # 词表构建
    num_words   = len(word2idx)
    word_onehot = word2onehot(num_words)

    lr            = 0.1
    epochs        = 10
    window_size   = 2 # 滑窗
    embedding_num = 200

    W1 = np.random.normal(size=(num_words,embedding_num))
    W2 = np.random.normal(size=(embedding_num,num_words))

    for epoch in range(epochs):
        for text in tqdm(corpus): # 每句话
            for idx, center_word in enumerate(text): # 每句话里面的每个词
                
                context_words      = text[idx-window_size : idx] + text[idx+1 : idx+1+window_size]
                center_word_onehot = word_onehot[word2idx[center_word]]
                
                for context_word in context_words:
                    
                    context_word_onehot = word_onehot[word2idx[context_word]]

                    hidden = context_word_onehot @ W1  #
                    pre    = hidden @ W2
                    p      = softmax(pre)
                    loss   = - np.sum(center_word_onehot * np.log(p))

                    dpre = G = p - center_word_onehot

                    dW2      = hidden.T @ G
                    d_hidden = G @ W2.T
                    dW1      = context_word_onehot.T @ d_hidden

                    W1 = W1 - lr * dW1
                    W2 = W2 - lr * dW2
        print(f"Epoch: {epoch}/{epochs}, Loss: {loss}")

    # 示例：获取词 "natural" 的词向量
    word = "不等式"
    word_vector = get_word_vector(word)
    print(f'Word vector for {word}: {word_vector}')

# Word2Vec - skip-gram 模型: 中心词预测周围词
# 数学数据集：data/word2vec
# 负采样


import os
import torch
import jieba
import random

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

    
# 生成正、负样本合集
def get_triple(center_word_idx, text):
    center_word   = text[center_word_idx]
    context_words = text[center_word_idx-window_size : center_word_idx] \
                  + text[center_word_idx+1 : center_word_idx+1+window_size]

    res = [(center_word,i,np.array([[1]])) for i in context_words]

    for i in range(num_neg_samples):
        neg_sample = random.choice(idx2word)

        if neg_sample in context_words or neg_sample == center_word:
            continue
        res.append((center_word, context_word, np.array([[0]])))

    return res

def sigmoid(x):
    x = np.clip(x, -30, 30)
    return 1 / (1 + np.exp(-x))


if __name__ == "__main__":
    corpus = read_data(osp.join("data","word2vec","数学原始数据.csv"))
    
    word2idx, idx2word = word2index(corpus) # 词表构建
    num_words   = len(word2idx)
    word_onehot = word2onehot(num_words)


    lr            = 0.07
    epochs        = 10
    num_neg_samples   = 10
    window_size   = 2 # 滑窗
    embedding_num = 300
    device = "cuda" if torch.cuda.is_available() else "cpu"
    

    W1 = np.random.normal(size=(num_words,embedding_num))
    W2 = np.random.normal(size=(embedding_num,num_words))


    for epoch in range(epochs):
        for text in tqdm(corpus): # batch_size=1
            for center_word_idx, center_word in enumerate(text):
                
                triple = get_triple(center_word_idx, text)

                for center_word, context_word, label in triple:
                    center_word_onehot  = word_onehot[word2idx[center_word]]
                    context_word_onehot = word_onehot[word2idx[context_word]]

                    # hidden = center_word_onehot @ W1
                    hidden = 1*W1[word2idx[center_word],None]

                    pre  = hidden @ W2[:,word2idx[context_word],None]
                    p    = sigmoid(pre)
                    loss = -np.sum( label*np.log(p) + (1-label)*np.log(1-p) )

                    G        = p - label
                    delta_W2 = hidden.T @ G
                    delta_h  = G @ W2[:,word2idx[context_word],None].T
                    delta_W1 =  delta_h

                    # W1 -= lr * delta_W1
                    W1[   word2idx[center_word], None]  -= lr * delta_W1
                    W2[:, word2idx[context_word], None] -= lr * delta_W2

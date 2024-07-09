import os
import pandas as pd
import jieba
import numpy as np
from tqdm import tqdm

def get_data(file):
    all_data = pd.read_csv(file,encoding="gbk",names=["data"])
    all_data = all_data["data"].tolist()

    cut_data = []
    for data in all_data:
        word_cut = jieba.lcut(data)
        cut_data.append(word_cut)

    return cut_data

def build_word_2_index(all_data):
    word_2_index = {}
    for data in all_data:
        for w in data:
            word_2_index[w] = word_2_index.get(w,len(word_2_index))
    return  word_2_index

def build_word_2_onehot(len_):
    return np.eye(len_).reshape(len_,1,len_)


def softmax(x):
    max_x = np.max(x,axis=-1)
    ex = np.exp(x-max_x)

    sum_ex = np.sum(ex,axis=1)

    result = ex / sum_ex

    return result


if __name__ == "__main__":
    all_data = get_data(os.path.join("data","word2vec","数学原始数据.csv"))
    word_2_index = build_word_2_index(all_data) # 词表构建
    words_len = len(word_2_index)
    word_2_onehot = build_word_2_onehot(words_len)

    epoch = 10
    n = 2 # 滑窗
    embedding_num = 200
    lr = 0.1

    w1 = np.random.normal(size=(words_len,embedding_num))
    w2 = np.random.normal(size=(embedding_num,words_len))

    for e in range(epoch):
        for words in tqdm(all_data): # 每句话

            for ni,now_word in enumerate(words): # 每句话里面的每个词
                other_words = words[ni-2:ni] + words[ni+1:ni+1+2]

                now_word_onehot = word_2_onehot[word_2_index[now_word]]
                for other_word in other_words:
                    other_word_onehot = word_2_onehot[word_2_index[other_word]]

                    hidden = other_word_onehot @ w1  #

                    pre = hidden @ w2

                    p = softmax(pre)
                    loss = - np.sum(now_word_onehot * np.log(p))

                    delta_pre = G = p - now_word_onehot

                    delta_w2 = hidden.T @ G
                    delta_hidden = G @ w2.T

                    delta_w1 = other_word_onehot.T @ delta_hidden

                    w1 = w1 - lr * delta_w1
                    w2 = w2 - lr * delta_w2

        print(loss)
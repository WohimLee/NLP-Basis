
import jieba

import numpy as np

import pandas as pd


def read_data(path):
    with open(path,encoding="utf-8") as f:
        all_data = f.read().split("\n")
    all_text = []
    all_label = []

    for data in all_data:
        data_s = data.split(" ")
        if len(data_s) != 2:
            continue
        text,lable = data_s

        all_text.append(text)
        all_label.append(int(lable))

    return all_text, all_label


def onehot(labels, class_num):
    n = len(labels)
    res = np.zeros((n, class_num))
    rows = np.arange(n)
    # cols = label.reshape(-1)
    # res[rows, cols] = 1
    res[rows, labels] = 1
    return res


def word2index(corpus):
    word2idx = {"PAD":0,"UNK":1}
    for text in corpus:
        for word in text:
            word2idx[word] = word2idx.get(word,len(word2idx))
    idx2word = list(word2idx)

    return word2idx, idx2word



def word2onehot(word_num):
    res = np.zeros((word_num, word_num))
    rows = np.arange(word_num)
    res[rows, rows] = 1
    return res



def load_stop_words(file = "stopwords.txt"):
    with open(file,"r",encoding = "utf-8") as f:
        return f.read().split("\n")

def cut_words(file="数学原始数据.csv"):
    stop_words = load_stop_words()
    result = []
    all_data = pd.read_csv(file,encoding = "gbk",names=["data"])["data"]
    for words in all_data:
        c_words = jieba.lcut(words)
        result.append([word for word in c_words if word not in stop_words])
    return result


def get_dict(data):
    index_2_word = []
    for words in data:
        for word in words:
            if word not in index_2_word:
                index_2_word.append(word)

    word_2_index = {word:index for index,word in enumerate(index_2_word)}
    word_size = len(word_2_index)

    word_2_onehot = {}
    for word,index in word_2_index.items():
        one_hot = np.zeros((1,word_size))
        one_hot[0,index] = 1
        word_2_onehot[word] = one_hot

    return word_2_index,index_2_word,word_2_onehot



def softmax_simple(x):
    ex = np.exp(x)
    return ex/np.sum(ex,axis = 1,keepdims = True)


def softmax(x):
    max_x = np.max(x,axis = -1,keepdims=True)
    x = x - max_x

    # x = np.clip(x, -1e10, 100)
    ex = np.exp(x)
    sum_ex = np.sum(ex, axis=1, keepdims=True)

    result = ex / sum_ex

    result = np.clip(result, 1e-10, 1e10)
    return result





def get_word_onehot(len_):
    onehot = np.zeros((len_,len_))

    for i in range(len(onehot)):
        onehot[i][i] = 1

    return onehot

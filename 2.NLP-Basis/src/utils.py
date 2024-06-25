
import jieba

import numpy as np

import pandas as pd



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

def make_onehot(labels, class_num):
    result = np.zeros((len(labels), class_num))

    for idx, cls in enumerate(labels):
        result[idx][cls] = 1
    return result


def get_word_2_index(all_text):
    word_2_index = {"PAD":0}
    for text in all_text:
        for w in text:
            word_2_index[w] = word_2_index.get(w,len(word_2_index))

    index_2_word = list(word_2_index)

    return word_2_index,index_2_word

def get_word_onehot(len_):
    onehot = np.zeros((len_,len_))

    for i in range(len(onehot)):
        onehot[i][i] = 1

    return onehot

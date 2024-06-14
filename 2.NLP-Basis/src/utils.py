
import jieba

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



def softmax(x):
    ex = np.exp(x)
    return ex/np.sum(ex,axis = 1,keepdims = True)
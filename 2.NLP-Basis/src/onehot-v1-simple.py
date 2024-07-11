# 用 One-Hot 做一个简单的文本/情感分类任务
# 自己构建数据集 data/test

import numpy as np
import os.path as osp

from torch.utils.data import Dataset, DataLoader
from utils import softmax


def read_data(path):
    with open(path,encoding="utf-8") as f:
        all_data = f.read().split("\n")
    corpus = []
    labels = []

    for data in all_data:
        data_s = data.split(" ")
        if len(data_s) != 2:
            continue
        text,lable = data_s

        corpus.append(text)
        labels.append(int(lable))

    return corpus, labels2onehot(labels, class_num)

def word2index(corpus):
    word2idx = {"PAD":0}
    for text in corpus:
        for word in text:
            word2idx[word] = word2idx.get(word, len(word2idx))

    idx2word = list(word2idx)

    return word2idx, idx2word
    
def words2onehot(words_num):
    res  = np.zeros((words_num, words_num))
    rows = np.arange(words_num)
    res[rows, rows] = 1
    return res
    
def labels2onehot(labels, class_num):
    result = np.zeros((len(labels), class_num))

    for idx, cls in enumerate(labels):
        result[idx][cls] = 1
    return result
    

class MyDataset(Dataset):
    def __init__(self, corpus, labels):
        self.corpus = corpus
        self.labels = labels

    def __getitem__(self, index):
        text  = self.corpus[index][:max_len]
        label = self.labels[index]

        wordsIdx = [word2idx[word] for word in text] # 每句话所有词的 idx

        wordsIdx = wordsIdx + [0] * (max_len - len(wordsIdx) ) # 不足 max_len 的补 0

        word_onehot = [words_onehot[wordIdx] for wordIdx in wordsIdx]

        word_onehot = np.array(word_onehot)

        return word_onehot, label

    def __len__(self):
        return  len(self.corpus)


if __name__ == "__main__":
    class_num = 2
    max_len = 8 # 每句话取的最大词数量
    
    
    train_path = osp.join("data", "test", "train.txt")  
    train_corpus, train_labels = read_data(train_path) # label 已经做 one-hot 处理
    
    word2idx, idx2word = word2index(train_corpus)
    num_words = len(word2idx) # 所有词的个数，one-hot 的编码维度
    words_onehot = words2onehot(num_words)
    
    batch_size = 1
    
    train_dataset = MyDataset(train_corpus, train_labels)
    train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    
    epochs = 10
    lr = 0.1
    
    W1 = np.random.normal(size=(num_words, 2))
    
    for epoch in range(epochs):
        for batch_word_onehot, batch_label in train_loader:
            
            batch_word_onehot = batch_word_onehot.numpy() # (batchSize, maxLen, onehotDim)
            batch_label       = batch_label.numpy()
            
            pre = batch_word_onehot @ W1 # (batchSize, maxLen, labelDim)
            pre_mean = np.mean(pre,axis=1) # (batchSize, labelDim)，因为我们要按整句话的意思去分类
            p = softmax(pre_mean)
            loss = -np.sum( batch_label * np.log(p) + (1-batch_label) * np.log(1-p))

            G = (p - batch_label) / len(pre)

            dpre = np.zeros_like(pre)
            for i in range(len(G)):
                for j in range(G.shape[1]):
                    dpre[i][:,j] = G[i][j]

            dW1 = batch_word_onehot.transpose(0,2,1) @ dpre

            dW1 = np.mean(dW1,axis=0)
            W1 = W1 - lr * dW1

        print(f"Epoch: {epoch}, Loss: {loss}")
    pass


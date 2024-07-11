# 用 One-Hot 做一个简单的文本/情感分类任务
# 清华文本分类数据集: data/tsinghua-news

import numpy as np
import os.path as osp

from torch.utils.data import Dataset, DataLoader
from utils import softmax


def read_data(path, num=None):
    with open(path, encoding="utf-8") as f:
        all_data = f.read().split("\n")
    corpus = []
    labels = []

    for data in all_data:
        data_s = data.split("\t")
        if len(data_s) != 2:
            continue
        text, label = data_s

        corpus.append(text)
        labels.append(int(label))

    if num and num > 0:
        return corpus[:num], labels[:num]
    elif num and num < 0 :
        return corpus[num:], labels[num:]
    else:
        return corpus, labels

class MyDataset(Dataset):
    def __init__(self, corpus, labels):
        self.corpus = corpus
        self.labels = labels

    def __getitem__(self, index):
        text  = self.corpus[index][:max_len]
        label = self.labels[index]

        wordsIdx = [word2idx.get(word, 1) for word in text] # 每句话所有词的 idx

        wordsIdx = wordsIdx + [0] * (max_len - len(wordsIdx) ) # 不足 max_len 的补 0

        word_onehot = [words_onehot[wordIdx] for wordIdx in wordsIdx]

        word_onehot = np.array(word_onehot)

        return word_onehot, label

    def __len__(self):
        return  len(self.corpus)

def word2index(corpus):
    word2idx = {"PAD":0, "UNK":1}
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

if __name__ == "__main__":
    class_num = 10
    max_len= 30

    train_corpus, train_labels = read_data(osp.join("data","tsinghua-news","train.txt"), 2000)
    test_corpus, test_labels   = read_data(osp.join("data","tsinghua-news","train.txt"), -300)

    train_labels = labels2onehot(train_labels, class_num)

    word2idx, idx2word = word2index(train_corpus)
    num_words = len(word2idx) # 所有词的个数，one-hot 的编码维度
    words_onehot = words2onehot(num_words)
    
    batch_size = 10

    train_dataset    = MyDataset(train_corpus, train_labels)
    train_dataloader = DataLoader(train_dataset,batch_size=batch_size, shuffle=True)

    test_dataset    = MyDataset(test_corpus, test_labels)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    epochs = 100

    lr = 0.2

    W1 = np.random.normal(size=(num_words,class_num))


    for epoch in range(epochs):
        for batch_word_onehot, batch_label in train_dataloader:
            
            batch_word_onehot = batch_word_onehot.numpy()
            batch_label       = batch_label.numpy()
            
            pre = batch_word_onehot @ W1
            pre_mean = np.mean(pre, axis=1)
            p = softmax(pre_mean)
            loss = - np.sum( batch_label * np.log(p) + (1-batch_label) * np.log(1-p) )

            G = (p-batch_label)/len(pre)

            dpre = np.zeros_like(pre)
            for i in range(len(G)):
                for j in range(G.shape[1]):
                    dpre[i][:,j] = G[i][j]

            dW1 = batch_word_onehot.transpose(0,2,1) @ dpre

            dW1 = np.mean(dW1,axis=0)
            W1 = W1 - lr * dW1

        right_num = 0
        for batch_word_onehot, batch_label in test_dataloader:
            batch_word_onehot = batch_word_onehot.numpy()
            batch_label       = batch_label.numpy()
            
            pre = batch_word_onehot @ W1
            pre_mean = np.mean(pre, axis=1)
            pre_mean = np.argmax(pre_mean, axis=-1)
            
            right_num += int(np.sum(pre_mean == batch_label))
            
        acc = right_num / len(test_dataset)
        print(f"Epoch: {epoch}, Loss: {loss}, Acc: {acc:.3f}")



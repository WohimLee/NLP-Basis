# 用 One-Hot 做一个简单的文本/情感分类任务
# 清华数据集: data/textCls

import numpy as np
import os.path as osp

from torch.utils.data import Dataset, DataLoader
from utils import onehot, word2onehot, softmax


class MyDataset(Dataset):
    def __init__(self, path, max_len):
        self.max_len = max_len
        self.corpus  = []
        self.labels  = []
        self.word2idx  = {"PAD":0,"UNK":1}
        self.idx2word  = {}
        
        self.read_data(path)
        self.word2index(self.corpus)
        
    def read_data(self, path):
        with open(path,encoding="utf-8") as f:
            all_data = f.read().split("\n")

        for data in all_data:
            data_s = data.split(" ")
            if len(data_s) != 2:
                continue
            text,lable = data_s

            self.corpus.append(text)
            self.labels.append(int(lable))
        self.labels = onehot(self.labels, 2) # 0 或 1，正面或者负面语句

    def word2index(self, corpus):
        for text in corpus:
            for word in text:
                self.word2idx[word] = self.word2idx.get(word,len(self.word2idx))
        self.idx2word = list(self.word2idx)
        
        self.onehot_dim = len(self.word2idx) # onehot 的维度，所有不重复词的总数
        self.words_onehot = word2onehot(self.onehot_dim)



    def __getitem__(self, index):
        text  = self.corpus[index][:max_len]
        label = self.labels[index]

        wordsIdx = [self.word2idx[word] for word in text] # 取出词的 idx

        wordsIdx = wordsIdx + [0] * (max_len - len(wordsIdx)) # 不够 max_len 的补 0

        word_onehot = [self.words_onehot[wordIdx] for wordIdx in wordsIdx]

        word_onehot = np.array(word_onehot)

        return word_onehot, label

    def __len__(self):
        return  len(self.corpus)



if __name__ == "__main__":
    path = osp.join("data", "test", "train.txt")
    
    max_len=8 # 每句话取的最大词数量
    batch_size = 1
    
    train_dataset = MyDataset(path, max_len)
    train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    
    epochs = 10
    lr = 0.1
    
    W1 = np.random.normal(size=(train_dataset.onehot_dim,2))
    
    for epoch in range(epochs):
        for batch_word_onehot, batch_label in train_loader:
            
            batch_word_onehot = batch_word_onehot.numpy() # (batchSize, maxLen, onehotDim)
            batch_label       = batch_label.numpy()
            
            pre = batch_word_onehot @ W1 # (batchSize, maxLen, labelDim)
            pre_mean = np.mean(pre,axis=1) # (batchSize, labelDim)
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

        print("Loss: ", loss)
    pass


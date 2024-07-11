
# 用 One-Hot 做一个简单的文本/情感分类任务
# 清华文本分类数据集: data/tsinghua-news
# 用 PyTorch 框架实现, 尝试使用 GPU

import os
import  torch

import numpy as np
import torch.nn as nn
import os.path as osp


from tqdm import tqdm
from torch.utils.data import  Dataset,DataLoader


def read_data(path, num=None):
    with open(path, encoding="utf-8") as f:
        all_data = f.read().split("\n")
    corpus = []
    labels = []

    for data in all_data:
        data_s = data.split("\t")
        if len(data_s) != 2:
            continue
        text, lable = data_s

        corpus.append(text)
        labels.append(int(lable))

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

        word_onehot = np.array(word_onehot, dtype=np.float32)

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


class Model(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.W1 = nn.Linear(onehot_dim, onehot_dim)
        self.W2 = nn.Linear(onehot_dim, class_num)
        self.loss_fun = nn.CrossEntropyLoss()

    def forward(self, x, label=None):
        pre = self.W1(x)
        pre = self.W2(pre)
        pre = torch.mean(pre,dim=1)

        if label is not None:
            loss = self.loss_fun(pre, label)
            return loss

        return torch.argmax(pre,dim=-1)


if __name__ == "__main__":
    class_num = 10
    max_len   = 30

    train_corpus, train_labels = read_data(osp.join("data","tsinghua-news","train.txt"), 2000)
    test_corpus, test_labels   = read_data(osp.join("data","tsinghua-news","train.txt"), -300)

    train_labels = labels2onehot(train_labels, class_num)

    word2idx, idx2word = word2index(train_corpus)
    num_words = onehot_dim = len(word2idx) # 所有词的个数，one-hot 的编码维度
    words_onehot = words2onehot(num_words)
    
    
    batch_size = 200
 
    train_dataset    = MyDataset(train_corpus, train_labels)
    train_dataloader = DataLoader(train_dataset,batch_size=batch_size, shuffle=True)

    test_dataset    = MyDataset(test_corpus, test_labels)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    lr     = 0.001
    epoch  = 100
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = Model().to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=lr)

    for e in range(epoch):
        for batch_word_onehot, batch_label in tqdm(train_dataloader):
            
            batch_word_onehot = batch_word_onehot.to(device)
            batch_label       = batch_label.to(device)

            loss = model(batch_word_onehot, batch_label)
            loss.backward()
            opt.step()
            opt.zero_grad()

        right_num = 0
        for batch_word_onehot,batch_label in test_dataloader:
            batch_word_onehot = batch_word_onehot.to(device)
            batch_label       = batch_label.to(device)
            pre = model(batch_word_onehot)
            right_num += int(torch.sum(pre == batch_label))

        acc = right_num / len(test_dataset)
        print(f"Epoch: {epoch}, Loss: {loss}, Acc: {acc:.3f}")
        
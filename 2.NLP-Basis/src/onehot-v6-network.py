
# 用 One-Hot 做一个简单的文本/情感分类任务
# 清华文本分类数据集: data/tsinghua-news
# 用 PyTorch 框架实现, 尝试使用 GPU
# 使用 PyTorch 自带的 nn.Embedding, 跟我们自己在 v5 实现的原理一样

import  torch

import numpy as np
import torch.nn as nn
import os.path as osp

from tqdm import tqdm
from torch.utils.data import  Dataset, DataLoader

def read_data(path,num=None):
    with open(path,encoding="utf-8") as f:
        all_data = f.read().split("\n")
    corpus = []
    labels = []

    for data in all_data:
        data_s = data.split("\t")
        if len(data_s) != 2:
            continue
        text,lable = data_s

        corpus.append(text)
        labels.append(int(lable))

    if num and num > 0:
        return corpus[:num],labels[:num]
    elif num and num < 0 :
        return corpus[num:], labels[num:]
    else:
        return corpus, labels

class MyDataset(Dataset):
    def __init__(self,corpus,labels):
        self.corpus = corpus
        self.labels = labels

    def __getitem__(self, index):
        text  = self.corpus[index][:max_len]
        label = self.labels[index]

        wordsIdx = [word2idx.get(word, 1) for word in text]
        wordsIdx = wordsIdx + [0] * (max_len - len(wordsIdx) )
        
        # wordvec = [wordvecs[i] for i in wordsIdx]
        # wordvec = np.array(wordvec, dtype=np.float32)

        return torch.tensor(wordsIdx), label


    def __len__(self):
        return  len(self.corpus)


def word2index(corpus):
    word2idx = {"PAD":0, "UNK":1}
    for text in corpus:
        for word in text:
            word2idx[word] = word2idx.get(word, len(word2idx))

    idx2word = list(word2idx)

    return word2idx, idx2word


def labels2onehot(labels, class_num):
    result = np.zeros((len(labels), class_num))

    for idx, cls in enumerate(labels):
        result[idx][cls] = 1
    return result


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        # self.w = nn.Linear(len(word_2_index),len(word_2_index))
        self.emb = nn.Embedding(num_words, vec_num) # 搜狗的字向量

        self.linear1  = nn.Linear(vec_num, 300)
        self.relu     = nn.ReLU()
        self.drop_out = nn.Dropout(0.1)
        self.cls      = nn.Linear(300,class_num)
        self.loss_fun = nn.CrossEntropyLoss()

    def forward(self, x, label=None):
        x = self.emb.forward(x) # (batch, max_len) ----> (batch, max_len, vec_num)
        x = self.linear1(x)     # (batch, max_len, 200) --> batch * seq_len * 300
        x = self.relu(x)    # batch * seq_len * 300 ---> batch * seq_len * 300
        x = self.drop_out(x)    # batch * seq_len * 300 ---> batch * seq_len * 300
        pre = self.cls(x)   # batch * seq_len * 300 --> batch * seq_len * 10
        pre = torch.max(pre,dim=1)[0] # batch * seq_len * 10 ----> batch * 10

        if label is not None:
            loss = self.loss_fun(pre,label)
            return loss

        return torch.argmax(pre,dim=-1)


if __name__ == "__main__":
    class_num = 10
    vec_num   = 200     # 词向量编码维度, embedding_num
    max_len   = 40      # 最大词，多了切，少了补

    train_corpus, train_labels = read_data(osp.join("data","tsinghua-news","train.txt"), 20000)
    test_corpus, test_labels   = read_data(osp.join("data","tsinghua-news","train.txt"), -300)

    train_labels = labels2onehot(train_labels, class_num)

    word2idx, idx2word = word2index(train_corpus)
    num_words = onehot_dim = len(word2idx) # 所有词的个数，one-hot 的编码维度
    
    # word_vec = get_word_onehot(len(word_2_index))
    # word_vec = get_word_random_vec(len(word_2_index))

    batch_size = 100

    train_dataset    = MyDataset(train_corpus, train_labels)
    train_dataloader = DataLoader(train_dataset,batch_size=batch_size, shuffle=True)

    test_dataset    = MyDataset(test_corpus, test_labels)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    lr     = 0.001
    epochs = 100
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = Model().to(device)
    opt   = torch.optim.Adam(model.parameters(),lr=lr)

    for epoch in range(epochs):
        for batch_wordemb, batch_label in tqdm(train_dataloader):
            batch_wordemb = batch_wordemb.to(device)
            batch_label   = batch_label.to(device)

            loss = model.forward(batch_wordemb, batch_label)
            loss.backward()
            opt.step()
            opt.zero_grad()

        right_num = 0
        for batch_wordemb, batch_label in test_dataloader:
            batch_wordemb = batch_wordemb.to(device)
            batch_label   = batch_label.to(device)
            pre = model.forward(batch_wordemb)
            right_num += int(torch.sum(pre == batch_label))

        acc = right_num / len(test_dataset)
        print(f"Epoch: {epoch}/{epochs}, Loss: {loss}, Acc: {acc:.3f}")
        
import os
import sys
import numpy as np


from torch.utils.data import  Dataset,DataLoader
from utils import make_onehot, softmax, get_word_2_index, get_word_onehot




def read_data(path,num=None):
    with open(path,encoding="utf-8") as f:
        all_data = f.read().split("\n")
    all_text = []
    all_label = []

    for data in all_data:
        data_s = data.split("\t")
        if len(data_s) != 2:
            continue
        text,lable = data_s

        all_text.append(text)
        all_label.append(int(lable))

    if num and num > 0:
        return all_text[:num],all_label[:num]
    elif num and num < 0 :
        return all_text[num:], all_label[num:]
    else:
        return all_text, all_label

class MyDataset(Dataset):
    def __init__(self,all_text,all_label):
        self.all_text = all_text
        self.all_label = all_label

    def __getitem__(self, index):
        text = self.all_text[index][:max_len]
        label = self.all_label[index]

        text_idx = [word_2_index.get(i,1) for i in text]

        text_idx = text_idx + [0] * (max_len - len(text_idx) )

        text_emb = [word_onehot[i] for i in text_idx]

        text_emb = np.array(text_emb)

        return text_emb,label

    def __len__(self):
        return  len(self.all_text)




if __name__ == "__main__":
    class_num = 10

    train_text, train_label = read_data(os.path.join("..","data","文本分类","train.txt"),2000)
    test_text, test_label = read_data(os.path.join("..","data","文本分类","train.txt"),-300)

    train_label = make_onehot(train_label,class_num)
    word_2_index, index_2_word = get_word_2_index(train_text)
    word_onehot = get_word_onehot(len(word_2_index))

    max_len= 30
    batch_size = 10
    epoch = 100

    lr = 0.2

    w1 = np.random.normal(size=(len(word_2_index),class_num))

    train_dataset = MyDataset(train_text,train_label)
    train_dataloader = DataLoader(train_dataset,batch_size = batch_size,shuffle=True)

    test_dataset = MyDataset(test_text, test_label)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    for e in range(epoch):
        for batch_text_emb,batch_label in train_dataloader:
            batch_text_emb = batch_text_emb.numpy()
            batch_label = batch_label.numpy()
            pre = batch_text_emb @ w1
            pre_mean = np.mean(pre,axis=1)
            p = softmax(pre_mean)

            loss = - np.sum( batch_label * np.log(p) + (1-batch_label) * np.log(1-p) )

            G = (p-batch_label)/len(pre)

            dpre = np.zeros_like(pre)
            for i in range(len(G)):
                for j in range(G.shape[1]):
                    dpre[i][:,j] = G[i][j]

            delta_w1 = batch_text_emb.transpose(0,2,1) @ dpre
            delta_w1 = np.mean(delta_w1,axis=0)
            w1 = w1 - lr * delta_w1

        right_num = 0
        for batch_text_emb, batch_label in test_dataloader:
            batch_text_emb = batch_text_emb.numpy()
            batch_label = batch_label.numpy()
            pre = batch_text_emb @ w1
            pre_mean = np.mean(pre, axis=1)

            pre_mean = np.argmax(pre_mean,axis=-1)
            right_num += int(np.sum(pre_mean == batch_label))
        acc = right_num / len(test_dataset)

        print(f"acc : {acc:.3f}")


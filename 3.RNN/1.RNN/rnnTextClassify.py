import torch
import torch.nn as nn
from torch.utils.data import  Dataset,DataLoader
import os
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


def get_word_2_index(all_text):
    word_2_index = {"PAD":0,"UNK":1}
    for text in all_text:
        for w in text:
            word_2_index[w] = word_2_index.get(w,len(word_2_index))

    index_2_word = list(word_2_index)

    return word_2_index,index_2_word

class MyDataset(Dataset):
    def __init__(self,all_text,all_label):
        self.all_text = all_text
        self.all_label = all_label

    def __getitem__(self, index):
        text = self.all_text[index][:max_len]
        label = self.all_label[index]

        text_idx = [word_2_index.get(i,1) for i in text]
        text_idx += [0] *( max_len- len(text_idx) )

        return torch.tensor(text_idx),torch.tensor(label)


    def __len__(self):
        return len(self.all_text)


class RnnTextCls(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(word_size,emb_num)
        self.rnn = nn.RNN(emb_num,rnn_hidden_num,batch_first=True)  # RNN : Linear , emb_num * 200
        self.cls = nn.Linear(rnn_hidden_num,class_num)
        self.loss_fun = nn.CrossEntropyLoss()

    def forward(self,x,lable=None):  #10 * 30  10 * 30 * 200 @ 200 * 400 = 10 * 30 * 400
        emb = self.emb.forward(x) # x : 10 * 30 --> 10 * 30 * emb_num
        rnn_out1,_ = self.rnn.forward(emb)
        # 10 * 30 * emb_num ---> 10 * 30 * hidden

        rnn_out1 = rnn_out1[:,-1] # 0 , -1 , mean , max
        pre = self.cls(rnn_out1)

        if lable is not None:
            loss = self.loss_fun(pre,lable)
            return loss
        else:
            return torch.argmax(pre,dim=-1)




if __name__ == "__main__":
    class_num = 10

    train_text, train_label = read_data(os.path.join("..","..","data","文本分类","train.txt"),20000)
    test_text, test_label = read_data(os.path.join("..","..","data","文本分类","test.txt"))

    word_2_index,index_2_word = get_word_2_index(train_text)

    epoch = 100
    batch_size = 10
    max_len = 30
    emb_num = 200
    rnn_hidden_num = 100
    lr = 0.00005
    word_size = len(word_2_index)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dataset = MyDataset(train_text,train_label)
    train_dataloader = DataLoader(train_dataset,batch_size=batch_size,shuffle=False)

    test_dataset = MyDataset(test_text, test_label)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = RnnTextCls().to(device)
    opt = torch.optim.Adam(model.parameters(),lr=lr)

    for e in range(epoch):
        model.train()
        for batch_text_idx,batch_lable in train_dataloader:
            batch_text_idx = batch_text_idx.to(device)
            batch_lable = batch_lable.to(device)

            loss = model.forward(batch_text_idx,batch_lable)
            loss.backward()

            opt.step()
            opt.zero_grad()

        model.eval()
        right_num = 0
        for batch_text_idx, batch_lable in test_dataloader:
            batch_text_idx = batch_text_idx.to(device)
            batch_lable = batch_lable.to(device)

            pre = model.forward(batch_text_idx)

            right_num += int(torch.sum(pre == batch_lable))

        acc = right_num / len(test_dataset)
        print(acc)
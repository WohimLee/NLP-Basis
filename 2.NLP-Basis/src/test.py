
import os
import numpy as np
import os.path as osp

from torch.utils.data import  Dataset, DataLoader

def read_data(path):
    with open(path, encoding='utf-8') as f:
        all_data = f.read().split("\n")
    
    all_text = []
    all_label = []
    
    for data in all_data:
        data_s = data.split(" ")
        if len(data_s) != 2:
            continue
        text, label = data_s
        
        all_text.append(text)
        all_label.append(int(label))
        
    return all_text, all_label



class MyDataset(Dataset):
    def __init__(self, all_text, all_label):
        self.all_text  = all_text
        self.all_label = all_label

    def __getitem__(self, index):
        text = self.all_text[index][:max_len]
        label = self.all_label[index]

        text_idx = [word_2_index[i] for i in text]

        text_idx = text_idx + [0] * (max_len - len(text_idx) )

        text_emb = [word_onehot[i] for i in text_idx]

        text_emb = np.array(text_emb)

        return text_emb,label

    def __len__(self):
        return  len(self.all_text)
        
if __name__ == "__main__":
    path = osp.join("data" ,"test/train.txt")
    train_text, train_label = read_data(path)
    
    batch_size = 2
    epoch = 10
    
    train_dataset = MyDataset(train_text,train_label)
    train_dataloader = DataLoader(train_dataset,batch_size = batch_size,shuffle=False)

    for e in range(epoch):
        for batch_text_emb, batch_label in train_dataloader:
            pass

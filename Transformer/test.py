import os
import torch
import torch.nn as nn
from torch.utils.data import  Dataset,DataLoader
from transformers import BertModel


def read_data(file_path,num=None):
    with open(file_path,"r",encoding="utf-8") as f:
        all_data = f.read().split("\n")

    all_text = []
    all_label = []

    for data in all_data:
        s_data = data.split("\t")
        if len(s_data) != 2:
            continue
        text,label = s_data
        all_text.append(text)
        all_label.append(int(label))
    if num:
        return all_text[:num],all_label[:num]
    else:
        return all_text, all_label

def get_word_2_index(all_data):
    word_2_index = {"[PAD]":0,"[UNK]":1}

    for data in all_data:
        for w in data:
            word_2_index[w] = word_2_index.get(w,len(word_2_index))
    return word_2_index

class MyDataset(Dataset):
    def __init__(self,all_text,all_label):
        self.all_text = all_text
        self.all_label = all_label


    def __len__(self):
        assert len(self.all_text) == len(self.all_label)
        return len(self.all_label)


    def __getitem__(self, index):

        text = self.all_text[index][:512]
        label = self.all_label[index]

        text_index = [word_2_index.get(i,1) for i in text]

        return text_index,label,len(text_index)

def coll_fn(batch_data):

    batch_text_idx,batch_label,batch_len = zip(*batch_data)
    batch_max_len = max(batch_len)

    batch_text_idx_new = []

    for text_idx,label,len_ in zip(batch_text_idx,batch_label,batch_len):
        text_idx = text_idx + [0] * (batch_max_len-len_)
        batch_text_idx_new.append(text_idx)

    return torch.tensor(batch_text_idx_new),torch.tensor(batch_label),torch.tensor(batch_len)

class Positional(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, x):

        return self.embedding(x)

class Multi_Head_attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(embedding_dim,embedding_dim)
        # self.tanh = nn.Tanh()
        self.gelu = nn.GELU()

    def forward(self, x):
        return self.gelu(self.linear(x))

class Norm(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        return self.norm(x)

class Feed_Forward(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(embedding_dim,feed_num)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(feed_num,embedding_dim)
        self.norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        linear1_out = self.linear1(x)
        gelu_out = self.gelu(linear1_out)
        linear2_out = self.linear2(gelu_out)
        norm_out = self.norm(linear2_out)
        return self.dropout(norm_out)

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.multi_head_attention = Multi_Head_attention()
        self.norm1 = Norm()
        self.feed_forward = Feed_Forward()
        self.norm2 = Norm()

    def forward(self,x):

        att_x = self.multi_head_attention.forward(x)
        norm1 = self.norm1.forward(att_x)

        adn_out1 = x + norm1

        ff_out = self.feed_forward.forward(adn_out1)
        norm2 = self.norm2.forward(ff_out)

        adn_out2 = adn_out1 + norm2

        return adn_out2

class MyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size,embedding_dim)
        self.positional_layer = Positional()

        self.encoder = nn.Sequential(*[Block() for _ in range(num_hidden_layers)])

    def forward(self,inputs):

        input_emb= self.embedding(inputs)
        position_emb = self.positional_layer(inputs)

        input_embeddings = input_emb + position_emb

        encoder_out = self.encoder.forward(input_embeddings)

        print("")


if __name__ == "__main__":
    bert = BertModel.from_pretrained("bert-base-chinese")
    train_text,train_label = read_data(os.path.join("..","data","文本分类","train.txt"),2000)
    test_text,test_label = read_data(os.path.join("..","data","文本分类","test.txt"))
    word_2_index = get_word_2_index(train_text)

    vocab_size = len(word_2_index)

    index_2_word = list(word_2_index)

    batch_size = 3
    epoch = 10
    embedding_dim = 768
    feed_num = 1024
    num_hidden_layers = 3

    train_dataset = MyDataset(train_text,train_label)
    train_dataloader = DataLoader(train_dataset,batch_size=batch_size,shuffle=False,collate_fn=coll_fn)

    model = MyTransformer()

    for e in range(epoch):
        for batch_text_idx,batch_label,batch_len in train_dataloader:

            model.forward(batch_text_idx)


    print("")






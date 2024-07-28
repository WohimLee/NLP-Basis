import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader

def get_data():
    english = ["apple","orange","banana","pear","black","red","white","pink","green","blue"]
    chinese = ["苹果","橙子","香蕉","梨","黑色","红色","白色","粉红色","绿色","蓝色"]

    return english,chinese

def get_word_2_index(english_data,chinese_data):
    eng_2_index = {"PAD":0,"UNK":1}
    chn_2_index = {"PAD":0,"UNK":1,"STA":2,"END":3}

    for eng in english_data:
        for w in eng:
            eng_2_index[w] = eng_2_index.get(w,len(eng_2_index))

    for chn in chinese_data:
        for w in chn:
            chn_2_index[w] = chn_2_index.get(w,len(chn_2_index))

    return eng_2_index,list(eng_2_index),chn_2_index,list(chn_2_index)

class TDataset(Dataset):
    def __init__(self,english_data,chinese_data):
        self.english = english_data
        self.chinese = chinese_data


    def __getitem__(self, index):
        e_data = self.english[index][:eng_max_len]
        c_data = self.chinese[index][:chn_max_len]

        e_index = [eng_2_index.get(i,1) for i in e_data] + [0] * (eng_max_len-len(e_data))
        c_index = [chn_2_index["STA"]] + [chn_2_index.get(i,1) for i in c_data] + [3] +  [0] * (chn_max_len-len(c_data))


        return torch.tensor(e_index),torch.tensor(c_index)


    def __len__(self):

        assert len(self.english) == len(self.chinese),"双语预料长度不一样~"
        return len(self.chinese)


class TModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.eng_embedding = nn.Embedding(len(eng_2_index),embedding_num)
        self.chn_embedding = nn.Embedding(len(chn_2_index),embedding_num)


        self.encoder = nn.RNN(embedding_num,hidden_size,batch_first=True,bidirectional=False)
        self.decoder = nn.RNN(embedding_num,hidden_size,batch_first=True,bidirectional=False)

        self.cls = nn.Linear(hidden_size,len(chn_2_index))
        self.loss_fun = nn.CrossEntropyLoss()

    def forward(self,eng_index,chn_index):

        eng_e = self.eng_embedding.forward(eng_index)
        chn_e = self.chn_embedding.forward(chn_index[:,:-1])
        batch_, seq_len = chn_e.shape[:2]

        _,encoder_out = self.encoder(eng_e)
        decoder_out,_ = self.decoder(chn_e,encoder_out)

        pre = self.cls(decoder_out)

        #  self.loss_fun(pre,chn_index[:,1:])

        loss = self.loss_fun(pre.reshape(batch_*seq_len,-1),chn_index[:,1:].reshape(-1))

        return loss

    def translate(self,eng_index):
        eng_e = self.eng_embedding.forward(eng_index)

        _,encoder_out = self.encoder(eng_e)
        decoder_h = encoder_out

        result = ""
        chn_index = chn_2_index["STA"]

        while True:
            chn_index = torch.tensor([[chn_index]],device=device)

            chn_e = self.chn_embedding(chn_index)
            decoder_out, decoder_h = self.decoder.forward(chn_e,decoder_h)
            pre = self.cls(decoder_out)
            chn_index = int(torch.argmax(pre,dim=-1))

            if len(result) > 20 or index_2_chn[chn_index] == "END":
                break

            result += index_2_chn[chn_index]

        return result



if __name__ == "__main__":
    english,chinese = get_data()
    eng_2_index,index_2_eng,chn_2_index,index_2_chn = get_word_2_index(english,chinese)

    epoch = 100
    batch_size = 1
    eng_max_len = 7
    chn_max_len = 3
    hidden_size = 100
    embedding_num = 50
    lr = 0.001

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dataset = TDataset(english,chinese)
    train_dataloader = DataLoader(train_dataset,batch_size=batch_size,shuffle=False)

    model = TModel().to(device)
    opt = torch.optim.Adam(model.parameters(),lr=lr)

    for e in range(epoch):
        for eng_index,chn_index in train_dataloader:
            eng_index = eng_index.to(device)
            chn_index = chn_index.to(device)

            loss = model.forward(eng_index,chn_index)
            loss.backward()

            opt.step()
            opt.zero_grad()

        print(f"loss:{loss:.3f}")
    while True:
        e_data = input("请输入:")
        e_index = [eng_2_index.get(i,1) for i in e_data] + [0] * (eng_max_len-len(e_data))
        e_index = torch.tensor([e_index],device=device)
        result = model.translate(e_index)
        print( "翻译结果:",result)
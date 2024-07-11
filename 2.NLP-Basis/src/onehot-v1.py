# 用 One-Hot 做一个简单的文本/情感分类任务
# 自己构建数据集 data/test


import os.path as osp


from utils import read_data, onehot, word2index, word2onehot






if __name__ == "__main__":
    path = osp.join("data", "test", "train.txt")
    train_text, train_label = read_data(path)
    train_label = onehot(train_label, 2) # 0 或 1，正面或者负面语句
    word2idx, idex2word = word2index(train_text)
    
    word_onehot = word2onehot(len(word2idx)) # 将输入的词转换成 onehot 编码
    pass


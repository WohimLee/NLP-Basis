import numpy as np
import re
from collections import Counter
import random

################# 数据准备 #################
# 示例文本数据
corpus = [
    "I love natural language processing",
    "Word2Vec is a technique for natural language processing",
    "Gensim is a library for topic modeling and document similarity",
    "Deep learning is useful for natural language understanding"
]

# 数据预处理：分词、去标点符号、小写化
def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    words = text.split()
    return words

corpus = [preprocess_text(doc) for doc in corpus]

# 构建词汇表
words = [word for doc in corpus for word in doc]
vocab = list(set(words))
vocab_size = len(vocab)

# 创建词与索引的映射
word_to_index = {word: i for i, word in enumerate(vocab)}
index_to_word = {i: word for i, word in enumerate(vocab)}

# 生成训练数据
def generate_training_data(corpus, window_size):
    training_data = []
    for sentence in corpus:
        sentence_length = len(sentence)
        for i, word in enumerate(sentence):
            context_words = []
            for j in range(max(0, i - window_size), min(sentence_length, i + window_size + 1)):
                if j != i:
                    context_words.append(sentence[j])
            for context_word in context_words:
                training_data.append((word, context_word))
    return training_data

window_size = 2
training_data = generate_training_data(corpus, window_size)

# 转换为索引
training_data = [(word_to_index[center], word_to_index[context]) for center, context in training_data]


################# 负采样 #################

# 定义负采样的数量
num_negative_samples = 5

# 生成负样本
def get_negative_samples(context_word, vocab_size, num_samples):
    negative_samples = []
    while len(negative_samples) < num_samples:
        negative_sample = random.randint(0, vocab_size - 1)
        if negative_sample != context_word:
            negative_samples.append(negative_sample)
    return negative_samples

################# 构建模型、训练 #################


# 定义超参数
embedding_dim = 100
learning_rate = 0.01
epochs = 1000

# 初始化权重
W1 = np.random.randn(vocab_size, embedding_dim)
W2 = np.random.randn(embedding_dim, vocab_size)

# 激活函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 前向传播
def forward(center_word):
    h = W1[center_word]
    u = np.dot(W2.T, h)
    y_pred = sigmoid(u)
    return y_pred, h, u

# 反向传播（更新参数）
def backward(center_word, context_word, negative_samples, y_pred, h):
    e = y_pred.copy()
    e[context_word] -= 1  # 正样本的误差

    # 更新 W2
    for k in negative_samples:
        e[k] += 1  # 负样本的误差
    dW2 = np.outer(h, e)
    W2 -= learning_rate * dW2

    # 更新 W1
    dW1 = np.dot(W2, e)
    W1[center_word] -= learning_rate * dW1

# 训练模型
for epoch in range(epochs):
    loss = 0
    for center_word, context_word in training_data:
        # 获取负样本
        negative_samples = get_negative_samples(context_word, vocab_size, num_negative_samples)
        
        # 前向传播
        y_pred, h, u = forward(center_word)
        
        # 计算损失
        loss -= np.log(sigmoid(u[context_word])) + sum(np.log(sigmoid(-u[negative_samples])))
        
        # 反向传播
        backward(center_word, context_word, negative_samples, y_pred, h)
    
    if epoch % 100 == 0:
        print(f'Epoch: {epoch}, Loss: {loss}')

# 获取词向量
def get_word_vector(word):
    word_index = word_to_index[word]
    return W1[word_index]

# 示例：获取词 "natural" 的词向量
word_vector = get_word_vector("natural")
print(f'Word vector for "natural": {word_vector}')

import numpy as np
import re
from collections import Counter

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
        for i, word in enumerate(sentence): # 每句话里面的每个词
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

# 定义超参数
embedding_dim = 100
learning_rate = 0.01
epochs = 1000

# 初始化权重
W1 = np.random.randn(vocab_size, embedding_dim)
W2 = np.random.randn(embedding_dim, vocab_size)

# 激活函数
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

# 前向传播
def forward(center_word):
    h = W1[center_word]
    u = np.dot(W2.T, h)
    y_pred = softmax(u)
    return y_pred, h, u

# 反向传播
def backward(center_word, context_word, y_pred, h):
    e = y_pred.copy()
    e[context_word] -= 1
    dW2 = np.outer(h, e)
    dW1 = np.outer(e, W2[:, center_word])
    W1[center_word] -= learning_rate * dW1
    W2 -= learning_rate * dW2

# 训练模型
for epoch in range(epochs):
    loss = 0
    for center_word, context_word in training_data:
        y_pred, h, u = forward(center_word)
        backward(center_word, context_word, y_pred, h)
        loss -= np.log(y_pred[context_word])
    if epoch % 100 == 0:
        print(f'Epoch: {epoch}, Loss: {loss}')

# 获取词向量
def get_word_vector(word):
    word_index = word_to_index[word]
    return W1[word_index]

# 示例：获取词 "natural" 的词向量
word_vector = get_word_vector("natural")
print(f'Word vector for "natural": {word_vector}')

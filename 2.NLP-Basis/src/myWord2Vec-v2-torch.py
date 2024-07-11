import torch
import torch.nn as nn
import torch.optim as optim




# 定义模型
class Word2Vec(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(Word2Vec, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(embedding_dim, vocab_size)
    
    def forward(self, inputs):
        embeds = self.embeddings(inputs)
        out = self.linear1(embeds)
        return out

# 训练模型
def train(data, model, loss_fn, optimizer, epochs=10):
    for epoch in range(epochs):
        total_loss = 0
        for context, target in data:
            context_idxs = torch.tensor([word_to_ix[w] for w in context], dtype=torch.long)
            model.zero_grad()
            log_probs = model(context_idxs)
            loss = loss_fn(log_probs, torch.tensor([word_to_ix[target]], dtype=torch.long))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch}, Loss: {total_loss}")



if __name__ == '__main__':
    # 超参数
    embedding_dim = 100
    context_size = 2  # 窗口大小为 2 个词（左右各一个）

    # 准备数据
    corpus = "We are about to study the idea of a computational process. Computational processes are abstract beings that inhabit computers. As they evolve, processes manipulate other abstract things called data. The evolution of a process is directed by a pattern of rules called a program. People create programs to direct processes. In effect, we conjure the spirits of the computer with our spells."
    words = corpus.split()
    vocab = set(words)
    word_to_ix = {word: i for i, word in enumerate(vocab)}
    ix_to_word = {i: word for i, word in enumerate(vocab)}

    data = []
    for i in range(context_size, len(words) - context_size):
        context = (
            [words[i - j - 1] for j in range(context_size)] +
            [words[i + j + 1] for j in range(context_size)]
        )
        target = words[i]
        data.append((context, target))
        
        
    # 初始化模型、损失函数和优化器
    model = Word2Vec(len(vocab), embedding_dim)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # 开始训练
    train(data, model, loss_fn, optimizer, epochs=100)

    # 输出一些词的嵌入向量
    word_embeddings = model.embeddings.weight.data.numpy()
    for word, idx in word_to_ix.items():
        print(f"Word: {word}, Embedding: {word_embeddings[idx]}")
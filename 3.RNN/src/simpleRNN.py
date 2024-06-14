
import torch
import torch.nn as nn

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        # i2h 和 i2o 是线性层，分别用于计算隐藏状态和输出
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        # 输入合并：将当前输入 input 和前一个时间步的隐藏状态 hidden 进行拼接
        combined = torch.cat((input, hidden), 1)
        # 计算新的隐藏状态：通过线性变换和激活函数计算新的隐藏状态
        hidden = self.i2h(combined)
        # 计算输出：通过线性变换和softmax激活函数计算输出
        output = self.i2o(hidden)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

# 参数设置
input_size = 10
hidden_size = 20
output_size = 5

rnn = SimpleRNN(input_size, hidden_size, output_size)

# 假设有一个长度为 T 的输入序列
T = 3
input_seq = [torch.randn(1, input_size) for _ in range(T)]
hidden = rnn.initHidden()

# 前向传播
for i in range(T):
    output, hidden = rnn(input_seq[i], hidden)
    print(f'Time step {i+1}: Output: {output}, Hidden: {hidden}')

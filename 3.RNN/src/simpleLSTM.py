import numpy as np

class LSTM:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Initialize weights
        self.W_f = np.random.randn(hidden_size, hidden_size + input_size)
        self.b_f = np.zeros((hidden_size, 1))

        self.W_i = np.random.randn(hidden_size, hidden_size + input_size)
        self.b_i = np.zeros((hidden_size, 1))

        self.W_C = np.random.randn(hidden_size, hidden_size + input_size)
        self.b_C = np.zeros((hidden_size, 1))

        self.W_o = np.random.randn(hidden_size, hidden_size + input_size)
        self.b_o = np.zeros((hidden_size, 1))

        self.W_y = np.random.randn(input_size, hidden_size)
        self.b_y = np.zeros((input_size, 1))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def tanh(self, x):
        return np.tanh(x)

    def forward(self, x, h_prev, C_prev):
        '''
        forward方法实现了LSTM的前向传播过程。它包括以下步骤：
        - 拼接输入和隐藏状态：将当前输入x_t和前一个时间步的隐藏状态h_{t-1}拼接在一起
        - 遗忘门：计算遗忘门的输出f_t
        - 输入门：计算输入门的输出i_t和候选细胞状态C_tilde
        - 更新细胞状态：结合遗忘门和输入门更新细胞状态C_t
        - 输出门：计算输出门的输出o_t和新的隐藏状态h_t
        - 计算输出：根据新的隐藏状态计算输出y_t
        '''
        
        # Concatenate hidden state and input
        combined = np.vstack((h_prev, x))

        # Forget gate
        f_t = self.sigmoid(np.dot(self.W_f, combined) + self.b_f)

        # Input gate
        i_t = self.sigmoid(np.dot(self.W_i, combined) + self.b_i)
        C_tilde = self.tanh(np.dot(self.W_C, combined) + self.b_C)

        # Cell state
        C_t = f_t * C_prev + i_t * C_tilde

        # Output gate
        o_t = self.sigmoid(np.dot(self.W_o, combined) + self.b_o)
        h_t = o_t * self.tanh(C_t)

        # Compute output
        y_t = np.dot(self.W_y, h_t) + self.b_y

        return y_t, h_t, C_t

# Example usage
input_size = 3
hidden_size = 5
seq_len = 10

lstm = LSTM(input_size, hidden_size)

# Random input sequence
x_seq = [np.random.randn(input_size, 1) for _ in range(seq_len)]

# Initial hidden and cell states
h_t = np.zeros((hidden_size, 1))
C_t = np.zeros((hidden_size, 1))

# Forward pass through the sequence
outputs = []
for x_t in x_seq:
    y_t, h_t, C_t = lstm.forward(x_t, h_t, C_t)
    outputs.append(y_t)

# Print the output for the last time step
print("Output of the last time step:", outputs[-1])

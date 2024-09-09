import torch
import torch.nn as nn
import math

class LayerNormalization(nn.Module):

    def __init__(self, features: int, eps:float=10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features)) # alpha is a learnable parameter
        self.bias = nn.Parameter(torch.zeros(features)) # bias is a learnable parameter

    def forward(self, x):
        # x: (batch, seq_len, hidden_size)
         # Keep the dimension for broadcasting
        mean = x.mean(dim = -1, keepdim = True) # (batch, seq_len, 1)
        # Keep the dimension for broadcasting
        std = x.std(dim = -1, keepdim = True) # (batch, seq_len, 1)
        # eps is to prevent dividing by zero or when std is very small
        return self.alpha * (x - mean) / (std + self.eps) + self.bias

class FeedForwardBlock(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) # w1 and b1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) # w2 and b2

    def forward(self, x):
        # (batch, seq_len, d_model) --> (batch, seq_len, d_ff) --> (batch, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))

class InputEmbeddings(nn.Module):

    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.d_model = d_model          # 512 向量维度
        self.vocab_size = vocab_size    # 词表大小
        self.embedding = nn.Embedding(vocab_size, d_model) # PyTorch 的Embedding

    def forward(self, x):
        # (batch, seq_len) --> (batch, seq_len, d_model)
        # Multiply by sqrt(d_model) to scale the embeddings according to the paper
        return self.embedding(x) * math.sqrt(self.d_model)
    
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        # Create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # Create a vector of shape (seq_len)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) # (seq_len, 1)
        # Create a vector of shape (d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # (d_model / 2)
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term) # sin(position * (10000 ** (2i / d_model))
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term) # cos(position * (10000 ** (2i / d_model))
        # Add a batch dimension to the positional encoding
        pe = pe.unsqueeze(0) # (1, seq_len, d_model)
        # Register the positional encoding as a buffer
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False) # (batch, seq_len, d_model)
        return self.dropout(x)

class ResidualConnection(nn.Module):
    
    def __init__(self, features: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(features)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x))) # 这里先 norm 再过 sublayer, 大多数实现是这样, 我们跟着学

class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        # 初始化函数，传入模型维度 d_model、注意力头数 h 和 dropout 率
        super().__init__()
        self.d_model = d_model # Embedding vector size 输入的特征维度（embedding 维度）
        self.h = h # Number of heads, 注意力头的数量
        # Make sure d_model is divisible by h, 确保 d_model 能够被 h 整除，这样每个头可以平分 d_model 维度
        assert d_model % h == 0, "d_model is not divisible by h"

        # 每个注意力头处理的维度大小 d_k
        self.d_k = d_model // h # Dimension of vector seen by each head, 每个头看到的向量维度
        self.w_q = nn.Linear(d_model, d_model, bias=False) # Wq
        self.w_k = nn.Linear(d_model, d_model, bias=False) # Wk
        self.w_v = nn.Linear(d_model, d_model, bias=False) # Wv
        
        # 用于多头注意力输出的线性变换矩阵 Wo
        self.w_o = nn.Linear(d_model, d_model, bias=False) # Wo
        # dropout 用于防止过拟合
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        """
        计算缩放点积注意力（Scaled Dot-Product Attention）
        
        参数：
        - query: 查询向量，形状 (batch, h, seq_len, d_k)
        - key: 键向量，形状 (batch, h, seq_len, d_k)
        - value: 值向量，形状 (batch, h, seq_len, d_k)
        - mask: 掩码，用于掩盖无关的位置
        - dropout: dropout 模块
        
        返回：
        - 加权后的值向量，形状 (batch, h, seq_len, d_k)
        - 注意力得分矩阵，形状 (batch, h, seq_len, seq_len)
        """
        # 获取向量维度 d_k
        d_k = query.shape[-1]
        
        # 1. 计算注意力得分：QK^T / sqrt(d_k)
        # Just apply the formula from the paper
        # (batch, h, seq_len, d_k) --> (batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        
        # 2. 应用掩码（mask），将无关的位置得分设为负无穷小（-1e9）
        if mask is not None:
            # Write a very low value (indicating -inf) to the positions where mask == 0
            attention_scores.masked_fill_(mask == 0, -1e9)
            
        # 3. 对注意力得分应用 softmax，转换为概率分布
        attention_scores = attention_scores.softmax(dim=-1) # (batch, h, seq_len, seq_len) # Apply softmax
        
        # 4. 应用 dropout（如果指定）
        if dropout is not None:
            attention_scores = dropout(attention_scores)
            
        # 5. 计算最终的加权值向量：Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
        # (batch, h, seq_len, seq_len) --> (batch, h, seq_len, d_k)
        # return attention scores which can be used for visualization
        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        """
        多头注意力机制的前向传播
        
        参数：
        - q: 查询向量，形状 (batch, seq_len, d_model)
        - k: 键向量，形状 (batch, seq_len, d_model)
        - v: 值向量，形状 (batch, seq_len, d_model)
        - mask: 掩码，用于掩盖特定位置
        
        返回：
        - 最终的多头注意力输出，形状 (batch, seq_len, d_model)
        """
        # 1. 通过线性变换生成查询、键和值向量
        query = self.w_q(q) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        key   = self.w_k(k) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        value = self.w_v(v) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)

        # 2. 将查询、键和值分别 reshape 成 (batch, seq_len, h, d_k)，再转置为 (batch, h, seq_len, d_k)
        # (batch, seq_len, d_model) --> (batch, seq_len, h, d_k) --> (batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key   = key.view(key.shape[0],     key.shape[1],   self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)
        
        # 3. 调用静态方法计算注意力机制，返回加权后的值和注意力得分
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)
        
        # 4. 将多头注意力的输出 reshape 回原始形状 (batch, seq_len, d_model)
        # Combine all the heads together
        # (batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # 5. 通过线性变换矩阵 Wo 将多头注意力的输出映射回 d_model 维度
        # Multiply by Wo
        # (batch, seq_len, d_model) --> (batch, seq_len, d_model)  
        return self.w_o(x)

class EncoderBlock(nn.Module):

    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        # 初始化函数，传入特征维度、self-attention模块、前馈网络模块和dropout率
        super().__init__()
        
        # 保存传入的 self-attention 模块和前馈网络模块
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        
        # 使用两个残差连接层，每个残差连接层包含 dropout 和 LayerNorm，用于稳定训练
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        # 首先对输入进行第一层残差连接，使用 self-attention 机制来处理输入
        # residual_connections[0] 包含 self-attention 机制的输出和输入 x
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        
        # 接下来进行第二层残差连接，使用前馈网络处理 self-attention 输出
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x    # 返回经过 self-attention 和前馈网络处理后的结果
    
class Encoder(nn.Module):

    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class DecoderBlock(nn.Module):

    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        """
        初始化解码器块 (Decoder Block)。
        
        参数：
        - features: 输入特征的维度大小
        - self_attention_block: 解码器的自注意力模块 (MultiHeadAttentionBlock)
        - cross_attention_block: 解码器与编码器之间的交叉注意力模块 (MultiHeadAttentionBlock)
        - feed_forward_block: 前馈神经网络模块 (FeedForwardBlock)
        - dropout: dropout 率，用于防止过拟合
        """
        super().__init__()
        
        # 解码器块的三个模块：自注意力、交叉注意力和前馈网络
        self.self_attention_block  = self_attention_block   # 自注意力模块
        self.cross_attention_block = cross_attention_block  # 交叉注意力模块（用于关注编码器输出）
        self.feed_forward_block    = feed_forward_block     # 前馈神经网络模块
        
        # 残差连接与 LayerNorm，解码器块包含 3 个残差连接
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        """
        解码器块的前向传播。

        参数：
        - x: 解码器的输入张量 (batch_size, tgt_seq_len, features)
        - encoder_output: 编码器的输出 (batch_size, src_seq_len, features)
        - src_mask: 源序列掩码，用于遮蔽编码器输入中的无效部分
        - tgt_mask: 目标序列掩码，用于遮蔽解码器的自注意力中的未来词

        返回：
        - 经过解码器块处理后的输出张量
        """
        # 1. 首先，应用自注意力模块，并通过残差连接将输入与自注意力的输出相加
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        # 2. 接下来，应用交叉注意力模块，关注编码器的输出，仍通过残差连接进行相加
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        # 3. 最后，应用前馈神经网络，并通过残差连接将输入与前馈网络的输出相加
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x
    
class Decoder(nn.Module):

    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)

class ProjectionLayer(nn.Module):

    def __init__(self, d_model, vocab_size) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x) -> None:
        # (batch, seq_len, d_model) --> (batch, seq_len, vocab_size)
        return self.proj(x)
    
class Transformer(nn.Module):

    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, tgt_embed: InputEmbeddings, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        # (batch, seq_len, d_model)
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
    
    def decode(self, encoder_output: torch.Tensor, src_mask: torch.Tensor, tgt: torch.Tensor, tgt_mask: torch.Tensor):
        # (batch, seq_len, d_model)
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    
    def project(self, x):
        # (batch, seq_len, vocab_size)
        return self.projection_layer(x)
    
def build_transformer(
        src_vocab_size: int, 
        tgt_vocab_size: int, 
        src_seq_len: int, 
        tgt_seq_len: int, 
        d_model: int=512, N: int=6, h: int=8, dropout: float=0.1, d_ff: int=2048
    ) -> Transformer:
    """
    构建一个完整的 Transformer 模型，包含编码器、解码器、嵌入层、位置编码和投影层。

    参数：
    - src_vocab_size: 源语言的词汇表大小
    - tgt_vocab_size: 目标语言的词汇表大小
    - src_seq_len: 源序列的最大长度
    - tgt_seq_len: 目标序列的最大长度
    - d_model: 模型维度，默认为 512
    - N: 编码器和解码器块的层数，默认为 6
    - h: 多头注意力机制中的头数，默认为 8
    - dropout: dropout 率，默认为 0.1
    - d_ff: 前馈神经网络中隐藏层的维度，默认为 2048

    返回：
    - transformer: 构建好的 Transformer 模型
    """
    
    # 1. 创建嵌入层，负责将输入的单词索引转换为特征向量
    src_embed = InputEmbeddings(d_model, src_vocab_size)    # 源语言嵌入层
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)    # 目标语言嵌入层

    # 2. 创建位置编码层，为输入添加位置信息
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout) # 源语言的位置编码
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout) # 目标语言的位置编码
    
    # 3. 构建编码器块列表，包含 N 层编码器，每层包括一个自注意力模块和一个前馈神经网络模块
    encoder_blocks = []
    for _ in range(N):
        # 每个编码器块包含一个多头自注意力模块和一个前馈神经网络模块
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(d_model, encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # 4. 构建解码器块列表，包含 N 层解码器，每层包括一个自注意力模块、一个交叉注意力模块和一个前馈神经网络模块
    decoder_blocks = []
    for _ in range(N):
        # 每个解码器块包含一个多头自注意力模块、一个多头交叉注意力模块和一个前馈神经网络模块
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(d_model, decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)
    
    # 5. 创建编/解码器，包含 N 个编/解码器块
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))
    
    # 6. 创建投影层，用于将解码器的输出投影到目标语言的词汇表空间
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)
    
    # 7. 构建完整的 Transformer 模型，将编码器、解码器、嵌入层、位置编码和投影层整合
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)
    
    # 8. 初始化 Transformer 模型的参数，使用 Xavier 均匀初始化，确保权重分布的良好初始化
    for p in transformer.parameters():
        if p.dim() > 1: # 只对多维参数进行初始化，跳过偏置等单维参数
            nn.init.xavier_uniform_(p)
    
    return transformer  # 返回构建的 Transformer 模型
# Attention

Self-Attention（自注意力）机制是 Transformer 模型的核心组件之一。它的主要作用是为输入序列中的每个位置生成一个新的表示，这个表示综合了序列中所有其他位置的信息。自注意力机制的强大之处在于，它能够直接建模序列中不同位置之间的关系，而不管它们之间的距离。

## 1 Self-Attention
自注意力机制的输入是一组查询（Query）、键（Key）和值（Value）向量。对于每一个输入位置（或时间步），我们首先生成一个查询向量，然后用这个查询向量去“关注”输入序列中的每一个位置，通过计算这个查询与各个键之间的相似度来决定如何组合各个位置的值。

<div align=center>
    <image src="imgs/scaled-dot-product-attention.png" width=200>
</div>

### 1.1 计算步骤

1. 计算查询 $(\mathrm{Q})$ 、键（K）和值（V） 向量：
- 给定输入向量 $x_i$ ，通过与权重矩阵 $W^Q 、 W^K$ 和 $W^V$ 相乘，分别得到查询向量 $q_i$ 、键向量 $k_i$ 和值向量 $v_i$ 。
$$
q_i=x_i W^Q, \quad k_i=x_i W^K, \quad v_i=x_i W^V
$$

2. 计算注意力权重
- 对于给定的查询向量 $q_i$, 计算它与每个键向量 $k_j$ 的点积, 然后将结果除以一个缩放因子 $\sqrt{d_k}$ （这里 $d_k$ 是键向量的维度），最后通过 softmax 函数计算得到权重。
    $$
    \operatorname{Attention}\left(q_i, k_j, v_j\right)=\operatorname{softmax}\left(\frac{q_i \cdot k_j}{\sqrt{d_k}}\right)
    $$

    其中，点积 $q_i \cdot k_j$ 衡量了查询与键之间的相似性， softmax 函数则将这些相似性转换为概率分布 （即注意力权重）

3. 计算加权和
- 将每个位置的值向量 $v_j$ 乘以其对应的注意力权重, 然后将所有值向量的加权和作为输出。
    $$
    \text { Output }_i=\sum_j \operatorname{Attention}\left(q_i, k_j, v_j\right) \times v_j
    $$

    这个输出向量代表了位置 $i$ 的新表示，它综合了输入序列中所有其他位置的信息，并且这些信息的贡献程度由注意力权重决定。


### 1.2 Self-Attention 的优势
- 捕捉长程依赖：相比于 RNN，Self-Attention 可以直接在一个步骤中建立输入序列中任意两个位置之间的联系，而不依赖于序列的长度。
- 计算效率高：由于每个位置的计算可以并行执行，Self-Attention 非常适合现代 GPU 的并行计算架构，这使得模型的训练和推理速度更快。
- 对齐机制的统一：在许多序列到序列任务（如机器翻译）中，Self-Attention 统一了输入和输出之间的对齐机制，使得模型可以更加灵活地处理复杂的依赖关系。

### 1.3 Self-Attention 的局限性
- 序列长度的平方复杂度：Self-Attention 需要计算每对输入位置之间的关系，因此其计算复杂度是输入序列长度的平方 $O\left(n^2\right)$ ，这在处理非常长的序列时可能成为瓶颈。
- 位置信息丟失：由于 Self-Attention 自身不具有序列顺序的概念，必须通过位置编码来显式添加位置信息。

## 2 Multi-Head Attention

Self-Attention 的一个扩展是 Multi-Head Attention。与单头注意力不同, 多头注意力机制使用多个并行的自注意力机制，每个"头"独立地执行一次自注意力计算，然后将这些头的输出连接起来，形成最终的表示。

多头注意力的公式如下:
$$
\operatorname{MultiHead}(Q, K, V)=\text { Concat }\left(\operatorname{head}_1, \operatorname{head}_2, \ldots, \operatorname{head}_h\right) W^O
$$

每个头的计算方式与单头注意力相同，只是它们使用不同的投影矩阵：
$$
\operatorname{head}_i=\operatorname{Attention}\left(Q W_i^Q, K W_i^K, V W_i^V\right)
$$

多头注意力的优势在于它可以让模型从不同的表示空间中获取信息，这样模型可以在多个子空间中独立关注不同的特征。最终，通过将这些头的输出连接起来，可以获得更丰富和多样的表示。




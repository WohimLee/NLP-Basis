
import torch
import torch.nn as nn

from .rnnBase import RNNBase
from torch import Tensor
from typing import List, Tuple, Optional, overload
 
from torch.nn.utils.rnn import PackedSequence
from torch import _VF

class RNN(RNNBase):

    @overload
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1,
                 nonlinearity: str = 'tanh', bias: bool = True, batch_first: bool = False,
                 dropout: float = 0., bidirectional: bool = False, device=None,
                 dtype=None): #  -> None
        """
        重载的构造函数，用于类型提示和静态分析，如果不需要，可以删除。
        
        参数:
        - input_size (int): 输入张量的特征维度大小。
        - hidden_size (int): 隐藏层张量的特征维度大小。
        - num_layers (int, 可选): RNN 层数，默认为 1。
        - nonlinearity (str, 可选): 非线性函数类型，默认为 'tanh'。可以是 'tanh' 或 'relu'。
        - bias (bool, 可选): 是否使用偏置，默认为 True。
        - batch_first (bool, 可选): 输入和输出张量的形状是否为 (batch, seq, feature)，默认为 False。
        - dropout (float, 可选): 除最后一层外的每一层的 dropout 概率，默认为 0.0。
        - bidirectional (bool, 可选): 是否为双向 RNN，默认为 False。
        - device (可选): 张量分配的设备。
        - dtype (可选): 张量的数据类型。
        """
        ...


    def __init__(self, *args, **kwargs):
        """
        实际的构造函数，处理参数并进行初始化。

        参数:
        - *args: 位置参数。
        - **kwargs: 关键字参数。
        """
        
        # 检查是否提供了 'proj_size' 参数，如果有则抛出错误
        if 'proj_size' in kwargs:
            raise ValueError("proj_size argument is only supported for LSTM, not RNN or GRU")
        
        # 提取 'nonlinearity' 参数，如果未提供则默认为 'tanh'
        self.nonlinearity = kwargs.pop('nonlinearity', 'tanh')
        # 根据 'nonlinearity' 参数设置模式
        if self.nonlinearity == 'tanh':
            mode = 'RNN_TANH'
        elif self.nonlinearity == 'relu':
            mode = 'RNN_RELU'
        else:
            # 如果提供了未知的非线性函数类型，则抛出错误
            raise ValueError(f"Unknown nonlinearity '{self.nonlinearity}'. Select from 'tanh' or 'relu'.")
        # 调用父类的构造函数，传递模式和其他参数
        super().__init__(mode, *args, **kwargs)

    @overload
    @torch._jit_internal._overload_method  # noqa: F811
    def forward(self, input: Tensor, hx: Optional[Tensor] = None): # -> Tuple[Tensor, Tensor]
        """
        提供类型提示的 `forward` 方法，用于当输入为 `Tensor` 时的情形。

        参数:
        - input (Tensor): 输入的张量。
        - hx (Optional[Tensor]): 隐藏状态张量，可选参数，默认为 `None`。

        返回:
        - Tuple[Tensor, Tensor]: 返回值为包含两个 `Tensor` 的元组。
        """
        pass # 用于静态分析，不产生实际所用，不需要可以删除

    @overload
    @torch._jit_internal._overload_method  # noqa: F811
    def forward(self, input: PackedSequence, hx: Optional[Tensor] = None): # -> Tuple[PackedSequence, Tensor]
        """
        提供类型提示的 `forward` 方法，用于当输入为 `PackedSequence` 时的情形。

        参数:
        - input (PackedSequence): 输入的 `PackedSequence`，用于处理变长序列。
        - hx (Optional[Tensor]): 隐藏状态张量，可选参数，默认为 `None`。

        返回:
        - Tuple[PackedSequence, Tensor]: 返回值为包含 `PackedSequence` 和 `Tensor` 的元组。
        """
        pass # 用于静态分析，不产生实际所用，不需要可以删除

    def forward(self, input, hx=None):  # noqa: F811
        """
        执行前向传播的方法。

        参数:
        - input (Tensor 或 PackedSequence): 输入数据。
        - hx (Optional[Tensor]): 初始的隐藏状态，默认为 None。

        返回:
        - Tuple[Tensor, Tensor]: 输出和新的隐藏状态。
        """
        self._update_flat_weights() # 调用 _update_flat_weights 方法以确保使用最新的权重张量

        num_directions = 2 if self.bidirectional else 1 # 确定方向数，双向RNN为2，否则为1
        orig_input = input

        # 如果输入是 PackedSequence 类型
        if isinstance(orig_input, PackedSequence):
            # 解包 PackedSequence
            input, batch_sizes, sorted_indices, unsorted_indices = input
            max_batch_size = batch_sizes[0]
            # script() is unhappy when max_batch_size is different type in cond branches, so we duplicate
            
            if hx is None:  # 如果 hx 为空，初始化 hx
                hx = torch.zeros(self.num_layers * num_directions,
                                 max_batch_size, self.hidden_size,
                                 dtype=input.dtype, device=input.device)
            else:           # 隐藏状态的每个批次应该与输入序列匹配
                # Each batch of the hidden state should match the input sequence that
                # the user believes he/she is passing in.
                hx = self.permute_hidden(hx, sorted_indices)
                
        # 输入不是 PackedSequence 类型
        else:
            batch_sizes = None
            if input.dim() not in (2, 3): # 检查输入张量的维度是否为 2D 或 3D
                raise ValueError(f"RNN: Expected input to be 2D or 3D, got {input.dim()}D tensor instead")
            
            # 判断输入是否是批量的
            is_batched = input.dim() == 3
            batch_dim = 0 if self.batch_first else 1
            
            # 如果输入不是批量的，调整维度
            if not is_batched:
                input = input.unsqueeze(batch_dim)
                if hx is not None:
                    if hx.dim() != 2:
                        raise RuntimeError(
                            f"For unbatched 2-D input, hx should also be 2-D but got {hx.dim()}-D tensor")
                    hx = hx.unsqueeze(1)
            else:
                if hx is not None and hx.dim() != 3:
                    raise RuntimeError(
                        f"For batched 3-D input, hx should also be 3-D but got {hx.dim()}-D tensor")
                    
            # 确定最大批量大小
            max_batch_size = input.size(0) if self.batch_first else input.size(1)
            sorted_indices = None
            unsorted_indices = None
            
            # 如果 hx 为空，初始化 hx
            if hx is None:
                hx = torch.zeros(self.num_layers * num_directions,
                                 max_batch_size, self.hidden_size,
                                 dtype=input.dtype, device=input.device)
            else:   # 隐藏状态的每个批次应该与输入序列匹配
                # Each batch of the hidden state should match the input sequence that
                # the user believes he/she is passing in.
                hx = self.permute_hidden(hx, sorted_indices)
                
        # 确保隐藏状态 hx 不为 None
        assert hx is not None
        
        # 检查前向传播的输入参数是否有效
        self.check_forward_args(input, hx, batch_sizes)
        
        # 确保 RNN 模式是 'RNN_TANH' 或 'RNN_RELU'
        assert self.mode == 'RNN_TANH' or self.mode == 'RNN_RELU'
        
        # 如果 batch_sizes 为空（即输入不是 PackedSequence 类型）
        if batch_sizes is None:
            if self.mode == 'RNN_TANH':
                # 调用 _VF.rnn_tanh 函数执行前向传播，使用 Tanh 激活函数
                result = _VF.rnn_tanh(input, hx, self._flat_weights, self.bias, self.num_layers,
                                      self.dropout, self.training, self.bidirectional,
                                      self.batch_first)
            else:
                # 调用 _VF.rnn_relu 函数执行前向传播，使用 ReLU 激活函数
                result = _VF.rnn_relu(input, hx, self._flat_weights, self.bias, self.num_layers,
                                      self.dropout, self.training, self.bidirectional,
                                      self.batch_first)
        else:
            # 如果 batch_sizes 不为空（即输入是 PackedSequence 类型）
            if self.mode == 'RNN_TANH':
                # 调用 _VF.rnn_tanh 函数执行前向传播，使用 Tanh 激活函数
                result = _VF.rnn_tanh(input, batch_sizes, hx, self._flat_weights, self.bias,
                                      self.num_layers, self.dropout, self.training,
                                      self.bidirectional)
            else:
                # 调用 _VF.rnn_relu 函数执行前向传播，使用 ReLU 激活函数
                result = _VF.rnn_relu(input, batch_sizes, hx, self._flat_weights, self.bias,
                                      self.num_layers, self.dropout, self.training,
                                      self.bidirectional)
        # 从结果中提取输出和隐藏状态
        output = result[0]
        hidden = result[1]
        # 如果原始输入是 PackedSequence 类型
        if isinstance(orig_input, PackedSequence):
            # 将输出重新打包为 PackedSequence
            output_packed = PackedSequence(output, batch_sizes, sorted_indices, unsorted_indices)
            # 返回重新打包的输出和排列后的隐藏状态
            return output_packed, self.permute_hidden(hidden, unsorted_indices)
        # 如果输入不是批量的
        if not is_batched:
            # 移除增加的维度
            output = output.squeeze(batch_dim)
            hidden = hidden.squeeze(1)
        # 返回输出和排列后的隐藏状态
        return output, self.permute_hidden(hidden, unsorted_indices)
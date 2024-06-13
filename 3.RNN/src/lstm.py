

import torch
import torch.nn as nn

from .rnnBase import RNNBase
from .rnnBase import _apply_permutation
from torch import Tensor
from typing import List, Tuple, Optional, overload
 
from torch.nn.utils.rnn import PackedSequence
from torch import _VF


class LSTM(RNNBase):
    
    @overload
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1, bias: bool = True,
                 batch_first: bool = False, dropout: float = 0., bidirectional: bool = False,
                 proj_size: int = 0, device=None, dtype=None): #  -> None
        ...

    @overload
    def __init__(self, *args, **kwargs):
        ...

    def __init__(self, *args, **kwargs):
        '''
        实际构造函数
        '''
        super().__init__('LSTM', *args, **kwargs)

    def get_expected_cell_size(self, input: Tensor, batch_sizes: Optional[Tensor]): #  -> Tuple[int, int, int]
        """
        计算预期的细胞状态大小。

        参数:
        - input (Tensor): 输入的张量。
        - batch_sizes (Optional[Tensor]): 可选的张量，包含每个时间步的批次大小（用于可变长度的批次）。

        返回:
        - Tuple[int, int, int]: 预期的细胞状态大小，格式为 (num_layers * num_directions, mini_batch, hidden_size)。
        """
        # 如果 batch_sizes 不为空，获取第一个时间步的批次大小作为 mini_batch
        if batch_sizes is not None:
            mini_batch = int(batch_sizes[0])
        else:
            # 否则，根据 batch_first 标志从 input 张量中确定 mini_batch 的大小
            mini_batch = input.size(0) if self.batch_first else input.size(1)
        # 如果 LSTM 是双向的，num_directions 设为 2，否则设为 1
        num_directions = 2 if self.bidirectional else 1
        # 计算预期的细胞状态大小
        expected_hidden_size = (self.num_layers * num_directions,
                                mini_batch, self.hidden_size)
        # 返回预期的细胞状态大小
        return expected_hidden_size


    def check_forward_args(self,  # type: ignore[override]
                           input: Tensor,
                           hidden: Tuple[Tensor, Tensor],
                           batch_sizes: Optional[Tensor],
                           ):
        """
        检查前向传播的输入参数是否有效。

        参数:
        - input (Tensor): 输入的张量。
        - hidden (Tuple[Tensor, Tensor]): 隐藏状态张量的元组，包括 (hidden_state, cell_state)。
        - batch_sizes (Optional[Tensor]): 可选的张量，包含每个时间步的批次大小（用于可变长度的批次）。
        """
        # 检查输入张量的有效性
        self.check_input(input, batch_sizes)
        # 获取预期的隐藏状态大小，并检查 hidden[0] 是否符合预期大小
        self.check_hidden_size(hidden[0], self.get_expected_hidden_size(input, batch_sizes),
                               'Expected hidden[0] size {}, got {}')
        # 获取预期的细胞状态大小，并检查 hidden[1] 是否符合预期大小
        self.check_hidden_size(hidden[1], self.get_expected_cell_size(input, batch_sizes),
                               'Expected hidden[1] size {}, got {}')


    def permute_hidden(self,  # type: ignore[override]
                       hx: Tuple[Tensor, Tensor], 
                       permutation: Optional[Tensor]
                       ): #  -> Tuple[Tensor, Tensor]
        """
        根据给定的排列对隐藏状态张量进行重新排列。

        参数:
        - hx (Tuple[Tensor, Tensor]): 隐藏状态张量的元组，包括 (hidden_state, cell_state)。
        - permutation (Optional[Tensor]): 可选的张量，包含重新排列的索引。

        返回:
        - Tuple[Tensor, Tensor]: 重新排列后的隐藏状态张量的元组。
        """
        if permutation is None: # 如果 permutation 为 None，直接返回原始的隐藏状态张量
            return hx
        # 使用 _apply_permutation 对 hidden_state 和 cell_state 进行排列，并返回新的排列后的隐藏状态张量元组
        return _apply_permutation(hx[0], permutation), _apply_permutation(hx[1], permutation)


    @overload  # type: ignore[override]
    @torch._jit_internal._overload_method  # noqa: F811
    def forward(self, input: Tensor, hx: Optional[Tuple[Tensor, Tensor]] = None
                ):  #  -> Tuple[Tensor, Tuple[Tensor, Tensor]] noqa: F811
        pass


    @overload
    @torch._jit_internal._overload_method  # noqa: F811
    def forward(self, input: PackedSequence, hx: Optional[Tuple[Tensor, Tensor]] = None
                ):  # -> Tuple[PackedSequence, Tuple[Tensor, Tensor]] noqa: F811
        pass

    def forward(self, input, hx=None):  # noqa: F811
        """
        执行前向传播的方法。

        参数:
        - input (Tensor 或 PackedSequence): 输入数据。
        - hx (Optional[Tuple[Tensor, Tensor]]): 初始的隐藏状态，包括 (hidden_state, cell_state)，默认为 None。

        返回:
        - Tuple[Tensor, Tuple[Tensor, Tensor]]: 输出和新的隐藏状态 (hidden_state, cell_state)。
        """
        # 更新扁平化的权重张量
        self._update_flat_weights()

        orig_input = input
        # xxx: isinstance check needs to be in conditional for TorchScript to compile
        batch_sizes = None
        do_permute = False
        num_directions = 2 if self.bidirectional else 1
        # 如果 proj_size 大于 0，real_hidden_size 使用 proj_size，否则使用 hidden_size
        real_hidden_size = self.proj_size if self.proj_size > 0 else self.hidden_size
        
        # 如果输入是 PackedSequence 类型
        if isinstance(orig_input, PackedSequence):
            input, batch_sizes, sorted_indices, unsorted_indices = input # 解包 PackedSequence
            max_batch_size = batch_sizes[0]
            # 如果 hx 为空，初始化 hx
            if hx is None:
                # 初始化 h_zeros 和 c_zeros
                h_zeros = torch.zeros(self.num_layers * num_directions,
                                      max_batch_size, real_hidden_size,
                                      dtype=input.dtype, device=input.device)
                c_zeros = torch.zeros(self.num_layers * num_directions,
                                      max_batch_size, self.hidden_size,
                                      dtype=input.dtype, device=input.device)
                hx = (h_zeros, c_zeros) 
            else:
                # Each batch of the hidden state should match the input sequence that
                # the user believes he/she is passing in.
                # 隐藏状态的每个批次应该与输入序列匹配
                hx = self.permute_hidden(hx, sorted_indices)
        else:
            # 检查输入张量的维度是否为 2D 或 3D
            if input.dim() not in (2, 3):
                raise ValueError(f"LSTM: Expected input to be 2D or 3D, got {input.dim()}D instead")
            # 判断输入是否是批量的
            is_batched = input.dim() == 3
            # 根据 batch_first 标志确定批量维度
            batch_dim = 0 if self.batch_first else 1
            
            # 如果输入不是批量的，增加批量维度
            if not is_batched:
                input = input.unsqueeze(batch_dim)
            # 确定最大批次大小
            max_batch_size = input.size(0) if self.batch_first else input.size(1)
            sorted_indices = None
            unsorted_indices = None
            
            # 如果 hx 为空，初始化 hx
            if hx is None:
                h_zeros = torch.zeros(self.num_layers * num_directions,
                                      max_batch_size, real_hidden_size,
                                      dtype=input.dtype, device=input.device)
                c_zeros = torch.zeros(self.num_layers * num_directions,
                                      max_batch_size, self.hidden_size,
                                      dtype=input.dtype, device=input.device)
                hx = (h_zeros, c_zeros)
            else:
                # 检查 hx 的维度是否与输入匹配
                if is_batched:
                    if (hx[0].dim() != 3 or hx[1].dim() != 3):
                        msg = ("For batched 3-D input, hx and cx should "
                               f"also be 3-D but got ({hx[0].dim()}-D, {hx[1].dim()}-D) tensors")
                        raise RuntimeError(msg)
                else:
                    if hx[0].dim() != 2 or hx[1].dim() != 2:
                        msg = ("For unbatched 2-D input, hx and cx should "
                               f"also be 2-D but got ({hx[0].dim()}-D, {hx[1].dim()}-D) tensors")
                        raise RuntimeError(msg)
                    # 增加批量维度
                    hx = (hx[0].unsqueeze(1), hx[1].unsqueeze(1))
                    
                # Each batch of the hidden state should match the input sequence that
                # the user believes he/she is passing in.
                # 每个批次的隐藏状态应该与输入序列匹配
                self.check_forward_args(input, hx, batch_sizes)
                hx = self.permute_hidden(hx, sorted_indices)
        # 如果 batch_sizes 为空（即输入不是 PackedSequence 类型）
        if batch_sizes is None:
            # 调用 _VF.lstm 函数执行前向传播
            result = _VF.lstm(input, hx, self._flat_weights, self.bias, self.num_layers,
                              self.dropout, self.training, self.bidirectional, self.batch_first)
        else:
            # 如果 batch_sizes 不为空（即输入是 PackedSequence 类型），调用 _VF.lstm 函数执行前向传播
            result = _VF.lstm(input, batch_sizes, hx, self._flat_weights, self.bias,
                              self.num_layers, self.dropout, self.training, self.bidirectional)
        # 从结果中提取输出和隐藏状态
        output = result[0]
        hidden = result[1:]
        # xxx: isinstance check needs to be in conditional for TorchScript to compile
        # 检查 orig_input 是否为 PackedSequence 类型，这个检查需要在条件语句中以便 TorchScript 编译
        if isinstance(orig_input, PackedSequence):
            # 将输出重新打包为 PackedSequence
            output_packed = PackedSequence(output, batch_sizes, sorted_indices, unsorted_indices)
            # 返回重新打包的输出和排列后的隐藏状态
            return output_packed, self.permute_hidden(hidden, unsorted_indices)
        else:
            # 如果输入不是批量的，移除增加的维度
            if not is_batched:
                output = output.squeeze(batch_dim)
                hidden = (hidden[0].squeeze(1), hidden[1].squeeze(1))
            # 返回输出和排列后的隐藏状态
            return output, self.permute_hidden(hidden, unsorted_indices)

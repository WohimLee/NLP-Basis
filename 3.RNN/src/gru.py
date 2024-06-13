
import torch
import torch.nn as nn

from .rnnBase import RNNBase
from torch import Tensor
from typing import List, Tuple, Optional, overload
 
from torch.nn.utils.rnn import PackedSequence
from torch import _VF

class GRU(RNNBase):
    """
    GRU 类，继承自 RNNBase，实现了门控循环单元（GRU）
    """

    @overload
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1, bias: bool = True,
                 batch_first: bool = False, dropout: float = 0., bidirectional: bool = False,
                 device=None, dtype=None): #  -> None
        ...

    @overload
    def __init__(self, *args, **kwargs):
        ...

    def __init__(self, *args, **kwargs):
        if 'proj_size' in kwargs:
            raise ValueError("proj_size argument is only supported for LSTM, not RNN or GRU")
        super().__init__('GRU', *args, **kwargs)

    @overload  # type: ignore[override]
    @torch._jit_internal._overload_method  # noqa: F811
    def forward(self, input: Tensor, hx: Optional[Tensor] = None):  # -> Tuple[Tensor, Tensor] noqa: F811
        pass

    @overload
    @torch._jit_internal._overload_method  # noqa: F811
    def forward(self, input: PackedSequence, hx: Optional[Tensor] = None):  # -> Tuple[PackedSequence, Tensor] noqa: F811
        pass

    def forward(self, input, hx=None):  # noqa: F811
        """
        执行前向传播的方法。

        参数:
        - input (Tensor 或 PackedSequence): 输入数据。
        - hx (Optional[Tensor]): 初始的隐藏状态，默认为 None。

        返回:
        - Tuple[Tensor, Tensor]: 输出和新的隐藏状态。
        """
        # 更新扁平化的权重张量
        self._update_flat_weights()

        orig_input = input
        # xxx: isinstance check needs to be in conditional for TorchScript to compile
        if isinstance(orig_input, PackedSequence):
            input, batch_sizes, sorted_indices, unsorted_indices = input # 解包 PackedSequence
            max_batch_size = batch_sizes[0]
            if hx is None:
                # 初始化 hx
                num_directions = 2 if self.bidirectional else 1
                hx = torch.zeros(self.num_layers * num_directions,
                                 max_batch_size, self.hidden_size,
                                 dtype=input.dtype, device=input.device)
            else:
                # Each batch of the hidden state should match the input sequence that
                # the user believes he/she is passing in.
                # 隐藏状态的每个批次应该与输入序列匹配
                hx = self.permute_hidden(hx, sorted_indices)
        else:
            batch_sizes = None
            # 检查输入张量的维度是否为 2D 或 3D
            if input.dim() not in (2, 3):
                raise ValueError(f"GRU: Expected input to be 2D or 3D, got {input.dim()}D instead")
            # 判断输入是否是批量的
            is_batched = input.dim() == 3
            # 根据 batch_first 标志确定批量维度
            batch_dim = 0 if self.batch_first else 1
            # 如果输入不是批量的，增加批量维度
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
            max_batch_size = input.size(0) if self.batch_first else input.size(1)
            sorted_indices = None
            unsorted_indices = None
            if hx is None:
                # 初始化 hx
                num_directions = 2 if self.bidirectional else 1
                hx = torch.zeros(self.num_layers * num_directions,
                                 max_batch_size, self.hidden_size,
                                 dtype=input.dtype, device=input.device)
            else:
                # Each batch of the hidden state should match the input sequence that
                # the user believes he/she is passing in.
                # 隐藏状态的每个批次应该与输入序列匹配
                hx = self.permute_hidden(hx, sorted_indices)
        # 检查前向传播的输入参数是否有效
        self.check_forward_args(input, hx, batch_sizes)
        if batch_sizes is None:
            # 调用 _VF.gru 函数执行前向传播
            result = _VF.gru(input, hx, self._flat_weights, self.bias, self.num_layers,
                             self.dropout, self.training, self.bidirectional, self.batch_first)
        else:
            # 如果 batch_sizes 不为空，表示输入是 PackedSequence 类型
            result = _VF.gru(input, batch_sizes, hx, self._flat_weights, self.bias,
                             self.num_layers, self.dropout, self.training, self.bidirectional)
        output = result[0]
        hidden = result[1]

        # xxx: isinstance check needs to be in conditional for TorchScript to compile
        if isinstance(orig_input, PackedSequence):
            # 将输出重新打包为 PackedSequence
            output_packed = PackedSequence(output, batch_sizes, sorted_indices, unsorted_indices)
            # 返回重新打包的输出和排列后的隐藏状态
            return output_packed, self.permute_hidden(hidden, unsorted_indices)
        else:
            # 如果输入不是批量的，移除增加的维度
            if not is_batched:
                output = output.squeeze(batch_dim)
                hidden = hidden.squeeze(1)
            # 返回输出和排列后的隐藏状态
            return output, self.permute_hidden(hidden, unsorted_indices)

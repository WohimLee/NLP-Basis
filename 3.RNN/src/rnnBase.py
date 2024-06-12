import math
import torch
import warnings
import weakref
import numbers
import torch.nn as nn

from torch import Tensor
from typing import List, Tuple, Optional, overload

def _apply_permutation(tensor: Tensor, permutation: Tensor, dim: int = 1): #  -> Tensor
    return tensor.index_select(dim, permutation)

class RNNBase(nn.Module):
    r"""Base class for RNN modules (RNN, LSTM, GRU).
    实现了RNN, LSTM和GRU类共享的部分，如模块初始化和参数存储管理的工具方法

    .. note::
    - RNNBase类没有实现forward方法
    - LSTM和GRU类覆盖了RNNBase实现的一些方法
    """
    # 定义类中的常量
    __constants__ = ['mode', 'input_size', 'hidden_size', 'num_layers', 'bias',
                     'batch_first', 'dropout', 'bidirectional', 'proj_size']
    __jit_unused_properties__ = ['all_weights']
    # 定义类属性的类型提示  
    mode: str
    input_size: int
    hidden_size: int
    num_layers: int
    bias: bool
    batch_first: bool
    dropout: float
    bidirectional: bool
    proj_size: int

    def __init__(self, mode: str, input_size: int, hidden_size: int,
                 num_layers: int = 1, bias: bool = True, batch_first: bool = False,
                 dropout: float = 0., bidirectional: bool = False, proj_size: int = 0,
                 device=None, dtype=None): #  -> None
        """
        初始化RNNBase类并设置指定参数

        参数:
            mode (str): RNN的类型（'LSTM', 'GRU', 'RNN_TANH', 'RNN_RELU'）。
            input_size (int): 输入特征的数量。
            hidden_size (int): 隐藏状态中的特征数量。
            num_layers (int, optional): 循环层数。默认为1。
            bias (bool, optional): 如果为False，则该层不使用偏置权重。默认为True。
            batch_first (bool, optional): 如果为True，则输入和输出张量的形状为(batch, seq, feature)。默认为False。
            dropout (float, optional): 如果非零，则在每个RNN层的输出上添加一个Dropout层，除了最后一层。默认为0。
            bidirectional (bool, optional): 如果为True，则变为双向RNN。默认为False。
            proj_size (int, optional): 如果大于0，则使用相应大小的LSTM投影。默认为0。
            device (torch.device, optional): 初始化模型参数的设备。默认为None。
            dtype (torch.dtype, optional): 模型参数的数据类型。默认为None。
        """
        factory_kwargs = {'device': device, 'dtype': dtype} # 将device和dtype参数打包到字典中，以便后续创建张量时使用
        super().__init__() # 调用父类（nn.Module）的初始化方法，确保RNNBase类正确继承nn.Module的功能
        self.mode = mode # RNN的类型（例如 'LSTM', 'GRU', 'RNN_TANH', 'RNN_RELU'）
        self.input_size = input_size    # 输入特征的数量
        self.hidden_size = hidden_size  # 隐藏状态的特征数量
        self.num_layers = num_layers    # RNN层的数量
        self.bias = bias                # 是否使用偏置
        self.batch_first = batch_first  # 输入和输出张量的形状是否为 (batch, seq, feature)
        self.dropout = float(dropout)   # Dropout概率，转换为浮点数
        self.bidirectional = bidirectional  # 是否为双向RNN
        self.proj_size = proj_size      # 投影大小
        
        # 初始化一个列表用于存储权重参数的弱引用，这里指定了列表中元素的类型为可选的弱引用参数
        self._flat_weight_refs: List[Optional[weakref.ReferenceType[nn.Parameter]]] = []
        
        num_directions = 2 if bidirectional else 1 # RNN的方向数量（双向为2，否则为1）
        # 验证dropout值
        if not isinstance(dropout, numbers.Number) or not 0 <= dropout <= 1 or \
                isinstance(dropout, bool):
            raise ValueError("dropout should be a number in range [0, 1] "
                             "representing the probability of an element being "
                             "zeroed")
        # 验证hidden_size和proj_size
        if dropout > 0 and num_layers == 1:
            warnings.warn("dropout option adds dropout after all but last "
                          "recurrent layer, so non-zero dropout expects "
                          f"num_layers greater than 1, but got dropout={dropout} and "
                          f"num_layers={num_layers}")

        if not isinstance(hidden_size, int):
            raise TypeError(f"hidden_size should be of type int, got: {type(hidden_size).__name__}")
        if hidden_size <= 0:
            raise ValueError("hidden_size must be greater than zero")
        if proj_size < 0:
            raise ValueError("proj_size should be a positive integer or zero to disable projections")
        if proj_size >= hidden_size:
            raise ValueError("proj_size has to be smaller than hidden_size")

        # 根据RNN模式确定门控大小
        if mode == 'LSTM':
            gate_size = 4 * hidden_size
        elif mode == 'GRU':
            gate_size = 3 * hidden_size
        elif mode == 'RNN_TANH':
            gate_size = hidden_size
        elif mode == 'RNN_RELU':
            gate_size = hidden_size
        else:
            raise ValueError("Unrecognized RNN mode: " + mode)

        self._flat_weights_names = []
        self._all_weights = []
        # 初始化每层和每个方向的权重
        for layer in range(num_layers):
            for direction in range(num_directions):
                # 如果proj_size大于0，real_hidden_size等于proj_size，否则等于hidden_size
                real_hidden_size = proj_size if proj_size > 0 else hidden_size
                # 如果是第一层，layer_input_size等于input_size，否则等于real_hidden_size乘以num_directions
                layer_input_size = input_size if layer == 0 else real_hidden_size * num_directions
                
                # 初始化输入到隐藏层的权重，形状为(gate_size, layer_input_size)
                w_ih = nn.Parameter(torch.empty((gate_size, layer_input_size), **factory_kwargs))
                # 初始化隐藏到隐藏层的权重，形状为(gate_size, real_hidden_size)
                w_hh = nn.Parameter(torch.empty((gate_size, real_hidden_size), **factory_kwargs))
                # 初始化输入到隐藏层的偏置，形状为(gate_size)
                b_ih = nn.Parameter(torch.empty(gate_size, **factory_kwargs))
                # 初始化隐藏到隐藏层的偏置，形状为(gate_size)
                # 第二个偏置向量用于CuDNN兼容。在标准定义中只需要一个偏置向量。
                b_hh = nn.Parameter(torch.empty(gate_size, **factory_kwargs))
                layer_params: Tuple[Tensor, ...] = ()
                if self.proj_size == 0:
                    if bias:    # 如果有偏置，包含w_ih, w_hh, b_ih, b_hh
                        layer_params = (w_ih, w_hh, b_ih, b_hh)
                    else:       # 如果没有偏置，只包含w_ih, w_hh
                        layer_params = (w_ih, w_hh)
                else: # 如果proj_size大于0，还需初始化投影权重，形状为(proj_size, hidden_size)
                    w_hr = nn.Parameter(torch.empty((proj_size, hidden_size), **factory_kwargs))
                    if bias: # 如果有偏置，包含w_ih, w_hh, b_ih, b_hh, w_hr
                        layer_params = (w_ih, w_hh, b_ih, b_hh, w_hr)
                    else: # 如果没有偏置，只包含w_ih, w_hh, w_hr
                        layer_params = (w_ih, w_hh, w_hr)

                # 如果是反向RNN，添加后缀'_reverse'
                suffix = '_reverse' if direction == 1 else ''
                # 参数名称模板
                param_names = ['weight_ih_l{}{}', 'weight_hh_l{}{}']
                if bias:
                    param_names += ['bias_ih_l{}{}', 'bias_hh_l{}{}']
                if self.proj_size > 0:
                    param_names += ['weight_hr_l{}{}']
                # 使用层号和方向后缀格式化参数名称
                param_names = [x.format(layer, suffix) for x in param_names]
                # 将参数名称与参数绑定
                for name, param in zip(param_names, layer_params):
                    setattr(self, name, param)
                # 将参数名称添加到扁平化权重名称列表
                self._flat_weights_names.extend(param_names)
                # 将参数名称添加到所有权重列表
                self._all_weights.append(param_names)

        self._init_flat_weights()

        self.reset_parameters()

    def _init_flat_weights(self):
        # 使用self._flat_weights_names中的每个权重名称，获取相应的权重并将它们存储在self._flat_weights列表中
        # 如果self中存在对应名称的属性，则获取该属性值（即权重）；否则，存储None
        self._flat_weights = [getattr(self, wn) if hasattr(self, wn) else None
                              for wn in self._flat_weights_names]
        # 使用self._flat_weights中的每个权重，创建弱引用并将它们存储在self._flat_weight_refs列表中
        # 如果权重不为None，则创建其弱引用；否则，存储None
        self._flat_weight_refs = [weakref.ref(w) if w is not None else None
                                  for w in self._flat_weights]
        # 调用flatten_parameters方法，处理参数的扁平化
        self.flatten_parameters()

    def __setattr__(self, attr, value):
        """
        设置对象的属性值。如果属性名在 _flat_weights_names 中，则更新 _flat_weights 中对应的值。

        参数:
            attr (str): 属性名称。
            value: 要设置的属性值。
        """
        # 检查当前对象是否有 _flat_weights_names 属性，并且attr是否在 _flat_weights_names 列表中
        if hasattr(self, "_flat_weights_names") and attr in self._flat_weights_names:
            # 获取attr在 _flat_weights_names 列表中的索引位置
            idx = self._flat_weights_names.index(attr)
            # 更新 _flat_weights 列表中对应索引处的值
            self._flat_weights[idx] = value
        super().__setattr__(attr, value)

    def flatten_parameters(self): #  -> None
        """
        重置参数数据指针，以便它们可以使用更快的代码路径。

        当前，这仅在模块位于GPU上并且启用了cuDNN时有效。
        否则，这个函数什么也不做。
        """
        # 如果 _flat_weights 仅部分实例化，则快速返回
        if len(self._flat_weights) != len(self._flat_weights_names):
            return
        # 如果 _flat_weights 中的任何一个不是 Tensor 类型，则快速返回
        for w in self._flat_weights:
            if not isinstance(w, Tensor):
                return

        # 如果 _flat_weights 中的任何张量不适用于 cuDNN 或张量的数据类型不同，则快速返回
        first_fw = self._flat_weights[0]    # 获取第一个权重
        dtype = first_fw.dtype              # 获取第一个权重的数据类型
        for fw in self._flat_weights:
            if (not isinstance(fw.data, Tensor) or not (fw.data.dtype == dtype) or
                    not fw.data.is_cuda or
                    not torch.backends.cudnn.is_acceptable(fw.data)):
                return

        # 如果任何参数别名，我们回退到较慢的复制代码路径。这是一个充分的检查，
        # 因为不完全别名的重叠参数缓冲区会破坏 Module.named_parameters() 中唯一性检查的假设
        unique_data_ptrs = {p.data_ptr() for p in self._flat_weights}
        if len(unique_data_ptrs) != len(self._flat_weights):
            return
        # 切换到第一个权重所在的CUDA设备
        with torch.cuda.device_of(first_fw):
            import torch.backends.cudnn.rnn as rnn

            # 注意：no_grad() 是必要的，因为 _cudnn_rnn_flatten_weight 是一个就地操作
            # 会影响 self._flat_weights
            with torch.no_grad():
                if torch._use_cudnn_rnn_flatten_weight():
                    num_weights = 4 if self.bias else 2
                    if self.proj_size > 0:
                        num_weights += 1
                    # 执行 cuDNN RNN 扁平化权重操作
                    torch._cudnn_rnn_flatten_weight(
                        self._flat_weights, num_weights,
                        self.input_size, rnn.get_cudnn_mode(self.mode),
                        self.hidden_size, self.proj_size, self.num_layers,
                        self.batch_first, bool(self.bidirectional))

    def _apply(self, fn, recurse=True):
        """
        对模型中的所有模块应用给定的函数fn。

        参数:
            fn (function): 应用于模块的函数。
            recurse (bool, optional): 是否递归地应用函数到子模块。默认为True。

        返回:
            返回应用函数后的结果。
        """
        ret = super()._apply(fn, recurse) # 调用父类的 _apply 方法，将函数 fn 应用于模型及其子模块

        # 重置 _flat_weights
        # 注意: 在移除此代码前要非常小心，因为第三方设备类型可能依赖此行为来正确处理 .to() 方法（例如LSTM）。
        self._init_flat_weights()

        return ret # 返回应用函数后的结果

    def reset_parameters(self): #  -> None
        # 计算标准差，用于均匀分布的范围
        stdv = 1.0 / math.sqrt(self.hidden_size) if self.hidden_size > 0 else 0
        # 遍历模型的所有参数
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)   # 使用均匀分布初始化参数，范围在 [-stdv, stdv] 之间

    def check_input(self, input: Tensor, batch_sizes: Optional[Tensor]): #  -> None
        if not torch.jit.is_scripting():
            if input.dtype != self._flat_weights[0].dtype and not torch._C._is_any_autocast_enabled():
                raise ValueError(f'input must have the type {self._flat_weights[0].dtype}, got type {input.dtype}')
        expected_input_dim = 2 if batch_sizes is not None else 3
        if input.dim() != expected_input_dim:
            raise RuntimeError(
                f'input must have {expected_input_dim} dimensions, got {input.dim()}')
        if self.input_size != input.size(-1):
            raise RuntimeError(
                f'input.size(-1) must be equal to input_size. Expected {self.input_size}, got {input.size(-1)}')

    def get_expected_hidden_size(self, input: Tensor, batch_sizes: Optional[Tensor]): #  -> Tuple[int, int, int]
        if batch_sizes is not None:
            mini_batch = int(batch_sizes[0])
        else:
            mini_batch = input.size(0) if self.batch_first else input.size(1)
        num_directions = 2 if self.bidirectional else 1
        if self.proj_size > 0:
            expected_hidden_size = (self.num_layers * num_directions,
                                    mini_batch, self.proj_size)
        else:
            expected_hidden_size = (self.num_layers * num_directions,
                                    mini_batch, self.hidden_size)
        return expected_hidden_size

    def check_hidden_size(self, hx: Tensor, expected_hidden_size: Tuple[int, int, int],
                          msg: str = 'Expected hidden size {}, got {}'): # -> None
        if hx.size() != expected_hidden_size:
            raise RuntimeError(msg.format(expected_hidden_size, list(hx.size())))

    def _weights_have_changed(self):
        # Returns True if the weight tensors have changed since the last forward pass.
        # This is the case when used with torch.func.functional_call(), for example.
        weights_changed = False
        for ref, name in zip(self._flat_weight_refs, self._flat_weights_names):
            weight = getattr(self, name) if hasattr(self, name) else None
            if weight is not None and ref is not None and ref() is not weight:
                weights_changed = True
                break
        return weights_changed

    def check_forward_args(self, input: Tensor, hidden: Tensor, batch_sizes: Optional[Tensor]):
        self.check_input(input, batch_sizes)
        expected_hidden_size = self.get_expected_hidden_size(input, batch_sizes)

        self.check_hidden_size(hidden, expected_hidden_size)

    def permute_hidden(self, hx: Tensor, permutation: Optional[Tensor]):
        if permutation is None:
            return hx
        return _apply_permutation(hx, permutation)


    def extra_repr(self): #  -> str
        s = '{input_size}, {hidden_size}'
        if self.proj_size != 0:
            s += ', proj_size={proj_size}'
        if self.num_layers != 1:
            s += ', num_layers={num_layers}'
        if self.bias is not True:
            s += ', bias={bias}'
        if self.batch_first is not False:
            s += ', batch_first={batch_first}'
        if self.dropout != 0:
            s += ', dropout={dropout}'
        if self.bidirectional is not False:
            s += ', bidirectional={bidirectional}'
        return s.format(**self.__dict__)

    def _update_flat_weights(self):
        if not torch.jit.is_scripting():
            if self._weights_have_changed():
                self._init_flat_weights()

    def __getstate__(self):
        # If weights have been changed, update the _flat_weights in __getstate__ here.
        self._update_flat_weights()
        # Don't serialize the weight references.
        state = self.__dict__.copy()
        del state['_flat_weight_refs']
        return state

    def __setstate__(self, d):
        super().__setstate__(d)
        if 'all_weights' in d:
            self._all_weights = d['all_weights']
        # In PyTorch 1.8 we added a proj_size member variable to LSTM.
        # LSTMs that were serialized via torch.save(module) before PyTorch 1.8
        # don't have it, so to preserve compatibility we set proj_size here.
        if 'proj_size' not in d:
            self.proj_size = 0

        if not isinstance(self._all_weights[0][0], str):
            num_layers = self.num_layers
            num_directions = 2 if self.bidirectional else 1
            self._flat_weights_names = []
            self._all_weights = []
            for layer in range(num_layers):
                for direction in range(num_directions):
                    suffix = '_reverse' if direction == 1 else ''
                    weights = ['weight_ih_l{}{}', 'weight_hh_l{}{}', 'bias_ih_l{}{}',
                               'bias_hh_l{}{}', 'weight_hr_l{}{}']
                    weights = [x.format(layer, suffix) for x in weights]
                    if self.bias:
                        if self.proj_size > 0:
                            self._all_weights += [weights]
                            self._flat_weights_names.extend(weights)
                        else:
                            self._all_weights += [weights[:4]]
                            self._flat_weights_names.extend(weights[:4])
                    else:
                        if self.proj_size > 0:
                            self._all_weights += [weights[:2]] + [weights[-1:]]
                            self._flat_weights_names.extend(weights[:2] + [weights[-1:]])
                        else:
                            self._all_weights += [weights[:2]]
                            self._flat_weights_names.extend(weights[:2])
            self._flat_weights = [getattr(self, wn) if hasattr(self, wn) else None
                                  for wn in self._flat_weights_names]

        self._flat_weight_refs = [weakref.ref(w) if w is not None else None
                                  for w in self._flat_weights]

    @property
    def all_weights(self): #  -> List[List[nn.Parameter]]
        return [[getattr(self, weight) for weight in weights] for weights in self._all_weights]

    def _replicate_for_data_parallel(self):
        replica = super()._replicate_for_data_parallel()
        # Need to copy these caches, otherwise the replica will share the same
        # flat weights list.
        replica._flat_weights = replica._flat_weights[:]
        replica._flat_weights_names = replica._flat_weights_names[:]
        return replica
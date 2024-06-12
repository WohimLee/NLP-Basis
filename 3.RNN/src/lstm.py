

import torch
import torch.nn as nn

from rnnBase import RNNBase
from rnnBase import _apply_permutation
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
        super().__init__('LSTM', *args, **kwargs)

    def get_expected_cell_size(self, input: Tensor, batch_sizes: Optional[Tensor]): #  -> Tuple[int, int, int]
        if batch_sizes is not None:
            mini_batch = int(batch_sizes[0])
        else:
            mini_batch = input.size(0) if self.batch_first else input.size(1)
        num_directions = 2 if self.bidirectional else 1
        expected_hidden_size = (self.num_layers * num_directions,
                                mini_batch, self.hidden_size)
        return expected_hidden_size

    # In the future, we should prevent mypy from applying contravariance rules here.
    # See torch/nn/modules/module.py::_forward_unimplemented
    def check_forward_args(self,  # type: ignore[override]
                           input: Tensor,
                           hidden: Tuple[Tensor, Tensor],
                           batch_sizes: Optional[Tensor],
                           ):
        self.check_input(input, batch_sizes)
        self.check_hidden_size(hidden[0], self.get_expected_hidden_size(input, batch_sizes),
                               'Expected hidden[0] size {}, got {}')
        self.check_hidden_size(hidden[1], self.get_expected_cell_size(input, batch_sizes),
                               'Expected hidden[1] size {}, got {}')

    # Same as above, see torch/nn/modules/module.py::_forward_unimplemented
    def permute_hidden(self,  # type: ignore[override]
                       hx: Tuple[Tensor, Tensor],
                       permutation: Optional[Tensor]
                       ): #  -> Tuple[Tensor, Tensor]
        if permutation is None:
            return hx
        return _apply_permutation(hx[0], permutation), _apply_permutation(hx[1], permutation)

    # Same as above, see torch/nn/modules/module.py::_forward_unimplemented
    @overload  # type: ignore[override]
    @torch._jit_internal._overload_method  # noqa: F811
    def forward(self, input: Tensor, hx: Optional[Tuple[Tensor, Tensor]] = None
                ):  #  -> Tuple[Tensor, Tuple[Tensor, Tensor]] noqa: F811
        pass

    # Same as above, see torch/nn/modules/module.py::_forward_unimplemented
    @overload
    @torch._jit_internal._overload_method  # noqa: F811
    def forward(self, input: PackedSequence, hx: Optional[Tuple[Tensor, Tensor]] = None
                ):  # -> Tuple[PackedSequence, Tuple[Tensor, Tensor]] noqa: F811
        pass

    def forward(self, input, hx=None):  # noqa: F811
        self._update_flat_weights()

        orig_input = input
        # xxx: isinstance check needs to be in conditional for TorchScript to compile
        batch_sizes = None
        do_permute = False
        num_directions = 2 if self.bidirectional else 1
        real_hidden_size = self.proj_size if self.proj_size > 0 else self.hidden_size
        if isinstance(orig_input, PackedSequence):
            input, batch_sizes, sorted_indices, unsorted_indices = input
            max_batch_size = batch_sizes[0]
            if hx is None:
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
                hx = self.permute_hidden(hx, sorted_indices)
        else:
            if input.dim() not in (2, 3):
                raise ValueError(f"LSTM: Expected input to be 2D or 3D, got {input.dim()}D instead")
            is_batched = input.dim() == 3
            batch_dim = 0 if self.batch_first else 1
            if not is_batched:
                input = input.unsqueeze(batch_dim)
            max_batch_size = input.size(0) if self.batch_first else input.size(1)
            sorted_indices = None
            unsorted_indices = None
            if hx is None:
                h_zeros = torch.zeros(self.num_layers * num_directions,
                                      max_batch_size, real_hidden_size,
                                      dtype=input.dtype, device=input.device)
                c_zeros = torch.zeros(self.num_layers * num_directions,
                                      max_batch_size, self.hidden_size,
                                      dtype=input.dtype, device=input.device)
                hx = (h_zeros, c_zeros)
            else:
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
                    hx = (hx[0].unsqueeze(1), hx[1].unsqueeze(1))
                # Each batch of the hidden state should match the input sequence that
                # the user believes he/she is passing in.
                self.check_forward_args(input, hx, batch_sizes)
                hx = self.permute_hidden(hx, sorted_indices)

        if batch_sizes is None:
            result = _VF.lstm(input, hx, self._flat_weights, self.bias, self.num_layers,
                              self.dropout, self.training, self.bidirectional, self.batch_first)
        else:
            result = _VF.lstm(input, batch_sizes, hx, self._flat_weights, self.bias,
                              self.num_layers, self.dropout, self.training, self.bidirectional)
        output = result[0]
        hidden = result[1:]
        # xxx: isinstance check needs to be in conditional for TorchScript to compile
        if isinstance(orig_input, PackedSequence):
            output_packed = PackedSequence(output, batch_sizes, sorted_indices, unsorted_indices)
            return output_packed, self.permute_hidden(hidden, unsorted_indices)
        else:
            if not is_batched:
                output = output.squeeze(batch_dim)
                hidden = (hidden[0].squeeze(1), hidden[1].squeeze(1))
            return output, self.permute_hidden(hidden, unsorted_indices)

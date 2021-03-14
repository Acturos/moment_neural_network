# -*- coding: utf-8 -*-
import torch
from torch import Tensor
from torch.nn.parameter import Parameter
from typing import Tuple


class MnnTempBn1d(torch.nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1,
                 track_running_stats: bool = True) -> None:
        super(MnnTempBn1d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.track_running_stats = track_running_stats
        self.weight = Parameter(torch.ones(num_features) * 2.5)
        self.register_buffer('running_var', torch.zeros(num_features))

    def _update_running_var(self, value):
        self.running_var = self.momentum * self.running_var + (1 - self.momentum) * value

    def forward(self, inp_u: Tensor, inp_s: Tensor, inp_r: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        if self.training:
            var_u = torch.var(inp_u, dim=0)
            self._update_running_var(var_u)

            weight = self.weight / torch.sqrt(var_u + self.eps)
            out_u = inp_u * weight
            out_s = inp_s * weight

        else:
            weight = self.weight / torch.sqrt(self.running_var + self.eps)
            out_u = inp_u * weight
            out_s = inp_s * weight

        return out_u, out_s, inp_r







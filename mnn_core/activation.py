# -*- coding: utf-8 -*-
import torch
import math
from typing import Tuple
from torch import Tensor
from .dawson_integration import fast_dawson_pytorch as fast_dawson
from .global_config import mnn_config


class MnnCoreModule(torch.nn.Module):
    __constants__ = ['L', 'vol_th', 'vol_rest', 'cut_off', 't_ref', 'boundary']
    L: float
    vol_th: float
    vol_rest: float
    cut_off: float
    t_ref: float
    boundary: float

    def __init__(self, conductance: float = mnn_config.get_value('conductance'), boundary: float = 9.,
                 vol_th: float = mnn_config.get_value('vol_th'), vol_rest: float = mnn_config.get_value('vol_rest'),
                 cut_off: float = mnn_config.get_value('cut_off'), t_ref: float = mnn_config.get_value('t_ref')):
        super(MnnCoreModule, self).__init__()
        self.dawson_module = fast_dawson.DoubleDawsonIntegrate(boundary=boundary)
        self.L = conductance
        self.vol_th = vol_th
        self.vol_rest = vol_rest
        self.cut_off = cut_off
        self.t_ref = t_ref
        self.boundary = boundary

    def _compute_bound(self, u: Tensor, s: Tensor) -> Tuple[Tensor, Tensor]:
        ub = (self.vol_th * self.L - u) / (math.sqrt(self.L) * s)
        lb = (self.vol_rest * self.L - u) / (math.sqrt(self.L) * s)
        return ub, lb

    def _region0_idx(self, u: Tensor, s: Tensor) -> Tuple[Tensor, Tensor]:
        idx0 = torch.gt(s, 0.)
        idx1 = torch.bitwise_and(idx0, torch.lt(self.vol_th * self.L - u, self.cut_off * math.sqrt(self.L) * s))
        idx0.bitwise_not_()
        return idx0, idx1

    def _auxiliary_func_mean(self, ub: Tensor, lb: Tensor, avoid_change: bool = True) -> Tensor:
        if avoid_change:
            temp = self.dawson_module.dawson1(ub.clone().detach()) - self.dawson_module.dawson1(lb.clone().detach())
            return 2 / self.L * temp
        else:
            return 2 / self.L * (self.dawson_module.dawson1(ub) - self.dawson_module.dawson1(lb))

    def forward_fast_mean(self, u: Tensor, s: Tensor) -> Tensor:
        idx0, idx1 = self._region0_idx(u, s)
        output = torch.zeros_like(u)

        ub, lb = self._compute_bound(u[idx1], s[idx1])
        output[idx1] = self._auxiliary_func_mean(ub, lb, avoid_change=False)

        idx1 = torch.bitwise_and(idx0, torch.gt(u, self.vol_th * self.L))
        output[idx1] = 1 / (self.t_ref - 1 / self.L * torch.log(1 - 1 / u[idx1]))
        return output

    def _auxiliary_func_fano(self, ub: Tensor, lb: Tensor, u_a: Tensor, avoid_change: bool = True) -> Tensor:
        if avoid_change:
            temp = self.dawson_module(ub.clone().detach()) - self.dawson_module(lb.clone().detach())
            return 8 / self.L / self.L * temp * torch.pow(u_a, 2)
        else:
            temp = self.dawson_module(ub) - self.dawson_module(lb)
            return 8 / self.L / self.L * temp * torch.pow(u_a, 2)

    def forward_fast_std(self, u: Tensor, s: Tensor, u_a: Tensor) -> Tensor:
        idx0, idx1 = self._region0_idx(u, s)
        output = torch.zeros_like(u)
        ub, lb = self._compute_bound(u[idx1], s[idx1])
        output[idx1] = self._auxiliary_func_fano(ub, lb, u_a[idx1], avoid_change=False)
        output[idx0] = torch.lt(u[idx0], 1)
        return output.mul_(u_a).sqrt_()

    def _auxiliary_func_chi(self, ub: Tensor, lb: Tensor, u_a: Tensor, s_a: Tensor, avoid_change: bool = True) -> Tensor:
        if avoid_change:
            delta_g = self.dawson_module.dawson1.func_dawson(ub.clone().detach()) -\
                      self.dawson_module.dawson1.func_dawson(lb.clone().detach())
            return torch.pow(u_a, 2) / s_a * delta_g * 2 / math.pow(self.L, 3/2)
        else:
            delta_g = self.dawson_module.dawson1.func_dawson(ub) - self.dawson_module.dawson1.func_dawson(lb)
            return torch.pow(u_a, 2) / s_a * delta_g * 2 / math.pow(self.L, 3 / 2)

    def forward_fast_chi(self, u: Tensor, s: Tensor, u_a: Tensor, s_a: Tensor) -> Tensor:
        idx0, idx1 = self._region0_idx(u, s)
        output = torch.zeros_like(u)
        ub, lb = self._compute_bound(u[idx1], s[idx1])
        output[idx1] = self._auxiliary_func_chi(ub, lb, u_a[idx1], s_a[idx1], avoid_change=False)

        idx1 = torch.bitwise_and(idx0, torch.gt(u, self.vol_th * self. L))
        output[idx1] = math.sqrt(2 / self.L) / torch.sqrt(self.t_ref - 1 / self.L * torch.log(1 - 1/u[idx1])) / \
            torch.sqrt(2 * u[idx1] - 1)

        return output

    def forward(self, u: Tensor, s: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        u_a = torch.zeros_like(u)
        s_a = torch.zeros_like(s)
        chi = torch.zeros_like(u)

        idx0, idx1 = self._region0_idx(u, s)
        ub, lb = self._compute_bound(u[idx1], s[idx1])
        u_a[idx1] = self._auxiliary_func_mean(ub, lb)

        idx2 = torch.bitwise_and(idx0, torch.gt(u, self.vol_th * self.L))
        u_a[idx2] = 1 / (self.t_ref - 1 / self.L * torch.log(1 - 1 / u[idx2]))

        s_a[idx1] = self._auxiliary_func_fano(ub, lb, u_a[idx1])
        s_a[idx0] = torch.lt(u[idx0], 1)
        s_a.mul_(u_a).sqrt_()

        chi[idx1] = self._auxiliary_func_chi(ub, lb, u_a[idx1], s_a[idx1])
        chi[idx2] = math.sqrt(2 / self.L) / torch.sqrt(self.t_ref - 1 / self.L * torch.log(1 - 1/u[idx1])) / \
            torch.sqrt(2 * u[idx1] - 1)
        return u_a, s_a, chi

    def _auxiliary_func_grad_uu_region0(self, ub: Tensor, lb: Tensor, s: Tensor, u_a: Tensor) -> Tensor:
        delta_g = self.dawson_module.dawson1.func_dawson(ub.clone().detach()) - \
                  self.dawson_module.dawson1.func_dawson(lb.clone().detach())
        return torch.pow(u_a, 2) / s * delta_g * 2 * math.pow(self.L, 3/2)

    def _auxiliary_func_grad_uu_region1(self, u, u_a):
        return self.vol_th * torch.pow(u_a, 2) / u / (u - self.vol_th * self.L)

    def _auxiliary_func_grad_us(self, ub, lb, s, u_a):
        temp = self.dawson_module.dawson1.func_dawson(ub.clone().detach()) * ub * \
            self.dawson_module.dawson1.func_dawson(lb.clone().detach()) * lb
        return torch.pow(u_a, 2) / s * temp / self.L

    def backward_fast_mean(self, u: Tensor, s: Tensor, u_a: Tensor) -> Tuple[Tensor, Tensor]:
        idx0, idx1 = self._region0_idx(u, s)
        grad_uu = torch.zeros_like(u)
        grad_us = torch.zeros_like(s)

        ub, lb = self._compute_bound(u[idx1], s[idx1])
        grad_uu[idx1] = self._auxiliary_func_grad_uu_region0(ub, lb, s[idx1], u_a[idx1])
        grad_us[idx1] = self._auxiliary_func_grad_us(ub, lb, s[idx1], u_a[idx1])

        idx1 = torch.bitwise_and(idx0, u > 1)
        grad_uu[idx1] = self._auxiliary_func_grad_uu_region1(u[idx1], u_a[idx1])

        return grad_uu, grad_us







# -*- coding: utf-8 -*-
import math
from typing import Tuple

import torch
from torch import Tensor

from .dawson_integration import fast_dawson_pytorch as dawson_module
from .global_config import mnn_config

torch.set_default_tensor_type(torch.DoubleTensor)


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
        self.dawson = dawson_module.DoubleDawsonIntegrate(boundary=boundary)
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
            temp = self.dawson.dawson1(ub.clone().detach()) - self.dawson.dawson1(lb.clone().detach())
            return 2 / self.L * temp
        else:
            return 2 / self.L * (self.dawson.dawson1(ub) - self.dawson.dawson1(lb))

    def forward_fast_mean(self, u: Tensor, s: Tensor) -> Tensor:
        idx0, idx1 = self._region0_idx(u, s)
        output = torch.zeros_like(u)

        ub, lb = self._compute_bound(u[idx1], s[idx1])
        output[idx1] = 1 / (self._auxiliary_func_mean(ub, lb, avoid_change=False) + self.t_ref)

        idx1 = torch.bitwise_and(idx0, torch.gt(u, self.vol_th * self.L))
        output[idx1] = 1 / (self.t_ref - 1 / self.L * torch.log(1 - 1 / u[idx1]))
        return output

    def _auxiliary_func_fano(self, ub: Tensor, lb: Tensor, u_a: Tensor, avoid_change: bool = True) -> Tensor:
        if avoid_change:
            temp = self.dawson(ub.clone().detach()) - self.dawson(lb.clone().detach())
            return 8 / self.L / self.L * temp * torch.pow(u_a, 2)
        else:
            temp = self.dawson(ub) - self.dawson(lb)
            return 8 / self.L / self.L * temp * torch.pow(u_a, 2)

    def forward_fast_std(self, u: Tensor, s: Tensor, u_a: Tensor) -> Tensor:
        idx0, idx1 = self._region0_idx(u, s)
        output = torch.zeros_like(u)
        ub, lb = self._compute_bound(u[idx1], s[idx1])
        output[idx1] = self._auxiliary_func_fano(ub, lb, u_a[idx1], avoid_change=False)
        output[idx0] = torch.lt(u[idx0], 1)
        return output.mul_(u_a).sqrt_()

    def _auxiliary_func_chi(self, ub: Tensor, lb: Tensor, u_a: Tensor, s_a: Tensor,
                            avoid_change: bool = True) -> Tensor:
        if avoid_change:
            delta_g = self.dawson.dawson1.func_dawson(ub.clone().detach()) - \
                      self.dawson.dawson1.func_dawson(lb.clone().detach())
            return torch.pow(u_a, 2) / s_a * delta_g * 2 / math.pow(self.L, 3 / 2)
        else:
            delta_g = self.dawson.dawson1.func_dawson(ub) - self.dawson.dawson1.func_dawson(lb)
            return torch.pow(u_a, 2) / s_a * delta_g * 2 / math.pow(self.L, 3 / 2)

    def forward_fast_chi(self, u: Tensor, s: Tensor, u_a: Tensor, s_a: Tensor) -> Tensor:
        idx0, idx1 = self._region0_idx(u, s)
        output = torch.zeros_like(u)
        ub, lb = self._compute_bound(u[idx1], s[idx1])
        output[idx1] = self._auxiliary_func_chi(ub, lb, u_a[idx1], s_a[idx1], avoid_change=False)

        idx1 = torch.bitwise_and(idx0, torch.gt(u, self.vol_th * self.L))
        output[idx1] = math.sqrt(2 / self.L) / torch.sqrt(self.t_ref - 1 / self.L * torch.log(1 - 1 / u[idx1])) / \
                       torch.sqrt(2 * u[idx1] - 1)

        return output

    def forward(self, u: Tensor, s: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        u_a = torch.zeros_like(u)
        s_a = torch.zeros_like(s)
        chi = torch.zeros_like(u)

        idx0, idx1 = self._region0_idx(u, s)
        ub, lb = self._compute_bound(u[idx1], s[idx1])
        u_a[idx1] = 1 / (self._auxiliary_func_mean(ub, lb) + self.t_ref)

        idx2 = torch.bitwise_and(idx0, torch.gt(u, self.vol_th * self.L))
        u_a[idx2] = 1 / (self.t_ref - 1 / self.L * torch.log(1 - 1 / u[idx2]))

        s_a[idx1] = self._auxiliary_func_fano(ub, lb, u_a[idx1])
        s_a[idx0] = torch.lt(u[idx0], 1) + 0.0
        s_a.mul_(u_a).sqrt_()

        chi[idx1] = self._auxiliary_func_chi(ub, lb, u_a[idx1], s_a[idx1])
        chi[idx2] = math.sqrt(2 / self.L) / torch.sqrt(self.t_ref - 1 / self.L * torch.log(1 - 1 / u[idx2])) / \
                    torch.sqrt(2 * u[idx2] - 1)
        return u_a, s_a, chi

    def _auxiliary_func_grad_uu_region0(self, ub: Tensor, lb: Tensor, s: Tensor, u_a: Tensor) -> Tensor:
        delta_g = self.dawson.dawson1.func_dawson(ub.clone().detach()) - \
                  self.dawson.dawson1.func_dawson(lb.clone().detach())
        return torch.pow(u_a, 2) / s * delta_g * 2 * math.pow(self.L, 3 / 2)

    def _auxiliary_func_grad_uu_region1(self, u, u_a):
        return self.vol_th * torch.pow(u_a, 2) / u / (u - self.vol_th * self.L)

    def _auxiliary_func_grad_us(self, ub, lb, s, u_a):
        temp = self.dawson.dawson1.func_dawson(ub.clone().detach()) * ub * \
               self.dawson.dawson1.func_dawson(lb.clone().detach()) * lb
        return torch.pow(u_a, 2) / s * temp / self.L

    def _auxiliary_func_delta_dawson1(self, ub: Tensor, lb: Tensor) -> Tuple[
        Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        delta_g_ub = self.dawson.dawson1.func_dawson(ub.clone().detach())
        delta_g_lb = self.dawson.dawson1.func_dawson(lb.clone().detach())
        delta_h_ub = self.dawson.func_dawson_2nd(ub.clone().detach())
        delta_h_lb = self.dawson.func_dawson_2nd(lb.clone().detach())
        delta_H_ub = self.dawson(ub.clone().detach())
        delta_H_lb = self.dawson(lb.clone().detach())
        return delta_g_ub, delta_g_lb, delta_h_ub, delta_h_lb, delta_H_ub, delta_H_lb

    @staticmethod
    def _auxiliary_func_delta_dawson2(delta_g_ub: Tensor, delta_g_lb: Tensor, delta_h_ub: Tensor, delta_h_lb: Tensor,
                                      delta_H_ub: Tensor, delta_H_lb: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        return delta_g_ub - delta_g_lb, delta_h_ub - delta_h_lb, delta_H_ub - delta_H_lb

    @staticmethod
    def _auxiliary_func_temp_dg(ub: Tensor, lb: Tensor, delta_g_ub: Tensor, delta_g_lb: Tensor) -> Tensor:
        return delta_g_ub * ub - delta_g_lb * lb

    @staticmethod
    def _auxiliary_func_temp_dh(ub: Tensor, lb: Tensor, delta_h_ub: Tensor, delta_h_lb: Tensor) -> Tensor:
        return delta_h_ub * ub - delta_h_lb * lb

    def _auxiliary_func_grad_s(self, ub: Tensor, lb: Tensor, s: Tensor, u_a: Tensor, s_a: Tensor) -> Tuple[
        Tensor, Tensor]:
        delta_g_ub, delta_g_lb, delta_h_ub, delta_h_lb, delta_H_ub, delta_H_lb = self._auxiliary_func_delta_dawson1(ub,
                                                                                                                    lb)
        delta_g, delta_h, delta_H = self._auxiliary_func_delta_dawson2(delta_g_ub, delta_g_lb, delta_h_ub, delta_h_lb,
                                                                       delta_H_ub, delta_H_lb)
        temp = s_a / s
        temp1 = (3 / self.L * u_a * delta_g - 0.5 * delta_h / delta_H) * math.sqrt(self.L) * temp

        temp_dg = self._auxiliary_func_temp_dg(ub, lb, delta_g_ub, delta_g_lb)
        temp_dh = self._auxiliary_func_temp_dh(ub, lb, delta_h_ub, delta_h_lb)
        temp2 = (3 / self.L * u_a * temp_dg - 0.5 * temp_dh / delta_H) * temp

        return temp1, temp2

    def _auxiliary_func_grad_ss(self, u: Tensor, u_a: Tensor) -> Tensor:
        return 1 / math.sqrt(2 * self.L) * torch.pow(u_a, 1.5) * torch.sqrt(
            1 / torch.pow(1 - u, 2) - 1 / torch.pow(u, 2))

    def _auxiliary_func_grad_chi(self, ub: Tensor, lb: Tensor, s: Tensor, u_a: Tensor, chi: Tensor) -> Tuple[
        Tensor, Tensor]:
        delta_g_ub, delta_g_lb, delta_h_ub, delta_h_lb, delta_H_ub, delta_H_lb = self._auxiliary_func_delta_dawson1(ub,
                                                                                                                    lb)
        delta_g, delta_h, delta_H = self._auxiliary_func_delta_dawson2(delta_g_ub, delta_g_lb, delta_h_ub, delta_h_lb,
                                                                       delta_H_ub, delta_H_lb)
        temp_dg = self._auxiliary_func_temp_dg(ub, lb, delta_g_ub, delta_g_lb)

        temp_u_a_2 = torch.pow(u_a, 2)
        grad_uu = temp_u_a_2 / s * delta_g * 2 * math.pow(self.L, 3 / 2)
        grad_us = temp_u_a_2 / s * temp_dg / self.L

        temp1 = chi / u_a
        temp2 = chi / s

        grad_chu = 0.5 * temp1 * grad_uu - math.sqrt(2.) / self.L * torch.sqrt(u_a / delta_H) * temp_dg / s \
                   + temp2 * delta_h / delta_H / 2. / math.sqrt(self.L)

        temp_dh = self._auxiliary_func_temp_dh(ub, lb, delta_h_ub, delta_h_lb)
        temp_dg = self._auxiliary_func_temp_dg(torch.pow(ub, 2), torch.pow(lb, 2), delta_g_ub, delta_g_lb) \
                  + self.vol_th * math.sqrt(self.L) / s

        grad_chs = 0.5 * temp1 * grad_us + (- temp_dg / delta_g + 0.5 / delta_H * temp_dh) * temp2

        return grad_chu, grad_chs

    def _auxiliary_func_grad_chu(self, u: Tensor, u_a: Tensor) -> Tensor:
        temp = self.vol_th * torch.pow(u_a, 2) / u / (u - self.vol_th * self.L)
        return 1 / math.sqrt(2 * self.L) / torch.sqrt(u_a * (2 * u - 1)) * temp - math.sqrt(2 / self.L) \
               / (self.vol_th * self.L) * torch.sqrt(u_a) * torch.pow(2 * u - 1, -1.5)

    def backward_fast_mean(self, u: Tensor, s: Tensor, u_a: Tensor) -> Tuple[Tensor, Tensor]:
        idx0, idx1 = self._region0_idx(u, s)
        grad_uu = torch.zeros_like(u)
        grad_us = torch.zeros_like(s)

        ub, lb = self._compute_bound(u[idx1], s[idx1])
        grad_uu[idx1] = self._auxiliary_func_grad_uu_region0(ub, lb, s[idx1], u_a[idx1])
        grad_us[idx1] = self._auxiliary_func_grad_us(ub, lb, s[idx1], u_a[idx1])

        idx1 = torch.bitwise_and(idx0, torch.gt(u, 1.))
        grad_uu[idx1] = self._auxiliary_func_grad_uu_region1(u[idx1], u_a[idx1])

        return grad_uu, grad_us

    def backward_fast_std(self, u: Tensor, s: Tensor, u_a: Tensor, s_a: Tensor) -> Tuple[Tensor, Tensor]:
        grad_su = torch.zeros_like(u)
        grad_ss = torch.zeros_like(s)

        idx0, idx1 = self._region0_idx(u, s)
        ub, lb = self._compute_bound(u[idx1], s[idx1])

        grad_su[idx1], grad_ss[idx1] = self._auxiliary_func_grad_s(ub, lb, s[idx1], u_a[idx1], s_a[idx1])

        idx1 = torch.bitwise_and(idx0, torch.gt(u, 1.))
        grad_ss[idx1] = self._auxiliary_func_grad_ss(u[idx1], u_a[idx1])

        return grad_su, grad_ss

    def backward_fast_chi(self, u: Tensor, s: Tensor, u_a: Tensor, chi: Tensor) -> Tuple[Tensor, Tensor]:

        idx0, idx1 = self._region0_idx(u, s)
        ub, lb = self._compute_bound(u[idx1], s[idx1])

        grad_chu = torch.zeros_like(u)
        grad_chs = torch.zeros_like(s)

        grad_chu[idx1], grad_chs[idx1] = self._auxiliary_func_grad_chi(ub, lb, s[idx1], u_a[idx1], chi[idx1])

        idx1 = torch.bitwise_and(idx0, torch.gt(u, 1.))
        grad_chu[idx1] = self._auxiliary_func_grad_chu(u[idx1], u_a[idx1])
        return grad_chu, grad_chs

    def _auxiliary_func_backward_region0(self, ub: Tensor, lb: Tensor, s: Tensor, u_a: Tensor, s_a: Tensor,
                                         chi: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        delta_g_ub, delta_g_lb, delta_h_ub, delta_h_lb, delta_H_ub, delta_H_lb = self._auxiliary_func_delta_dawson1(ub,
                                                                                                                    lb)
        delta_g, delta_h, delta_H = self._auxiliary_func_delta_dawson2(delta_g_ub, delta_g_lb, delta_h_ub, delta_h_lb,
                                                                       delta_H_ub, delta_H_lb)

        temp_u_a_2 = torch.pow(u_a, 2)
        grad_uu = temp_u_a_2 / s * delta_g * 2 * math.pow(self.L, 3 / 2)

        temp_dg = self._auxiliary_func_temp_dg(ub, lb, delta_g_ub, delta_g_lb)
        temp_dh = self._auxiliary_func_temp_dh(ub, lb, delta_h_ub, delta_h_lb)

        grad_us = temp_u_a_2 / s * temp_dg / self.L

        temp = s_a / s
        temp0 = 3 / self.L * u_a

        grad_su = (temp0 * delta_g - 0.5 * delta_h / delta_H) * math.sqrt(self.L) * temp
        grad_ss = (temp0 * temp_dg - 0.5 * temp_dh / delta_H) * temp

        temp1 = chi / u_a
        temp2 = chi / s

        grad_chu = 0.5 * temp1 * grad_uu - math.sqrt(2.) / self.L * torch.sqrt(u_a / delta_H) * temp_dg / s \
                   + temp2 * delta_h / delta_H / 2. / math.sqrt(self.L)

        temp_dg = self._auxiliary_func_temp_dg(torch.pow(ub, 2), torch.pow(lb, 2), delta_g_ub, delta_g_lb) \
                  + self.vol_th * math.sqrt(self.L) / s

        grad_chs = 0.5 * temp1 * grad_us + (- temp_dg / delta_g + 0.5 / delta_H * temp_dh) * temp2

        return grad_uu, grad_us, grad_su, grad_ss, grad_chu, grad_chs

    def _auxiliary_func_backward_region1(self, u: Tensor, u_a: Tensor) -> Tuple[Tensor, Tensor, Tensor]:

        grad_uu = self.vol_th * torch.pow(u_a, 2) / u / (u - self.vol_th * self.L)
        grad_ss = 1 / math.sqrt(2 * self.L) * torch.pow(u_a, 1.5) * torch.sqrt(
            1 / torch.pow(1 - u, 2) - 1 / torch.pow(u, 2))
        temp = self.vol_th * torch.pow(u_a, 2) / u / (u - self.vol_th * self.L)
        grad_chu = 1 / math.sqrt(2 * self.L) / torch.sqrt(u_a * (2 * u - 1)) * temp - math.sqrt(2 / self.L) \
                   / (self.vol_th * self.L) * torch.sqrt(u_a) * torch.pow(2 * u - 1, -1.5)

        return grad_uu, grad_ss, grad_chu

    def backward(self, u: Tensor, s: Tensor, u_a: Tensor, s_a: Tensor, chi: Tensor) -> Tuple[
        Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        idx0, idx1 = self._region0_idx(u, s)
        ub, lb = self._compute_bound(u[idx1], s[idx1])

        grad_uu = torch.zeros_like(u)
        grad_us = torch.zeros_like(u)

        grad_su = torch.zeros_like(s)
        grad_ss = torch.zeros_like(s)

        grad_chu = torch.zeros_like(chi)
        grad_chs = torch.zeros_like(chi)

        grad_uu[idx1], grad_us[idx1], grad_su[idx1], grad_ss[idx1], grad_chu[idx1], grad_chs[
            idx1] = self._auxiliary_func_backward_region0(
            ub, lb, s[idx1], u_a[idx1], s_a[idx1], chi[idx1])

        idx1 = torch.bitwise_and(idx0, torch.gt(u, 1.))
        grad_uu[idx1], grad_ss[idx1], grad_chu[idx1] = self._auxiliary_func_backward_region1(u[idx1], u_a[idx1])

        return grad_uu, grad_us, grad_su, grad_ss, grad_chu, grad_chs


mnn_core = MnnCoreModule()


class MnnActivationTrio(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u: Tensor, s: Tensor, rho: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        global mnn_core
        u_a, s_a, chi = mnn_core.forward(u, s)
        ctx.save_for_backward(u, s, rho, u_a, s_a, chi)

        if chi.dim() == 1:
            temp_func_chi = chi.reshape(1, -1)
            temp_func_chi = torch.mm(temp_func_chi.transpose(1, 0), temp_func_chi)
        # Multi sample case
        else:
            temp_func_chi = chi.reshape(chi.size()[0], 1, chi.size()[1])
            temp_func_chi = torch.bmm(temp_func_chi.transpose(-1, -2), temp_func_chi)
        corr_out = torch.mul(rho, temp_func_chi)
        if corr_out.dim() == 2:
            corr_out = corr_out.fill_diagonal_(1.0)

        else:
            torch.diagonal(corr_out, dim1=1, dim2=2).data.fill_(1.0)

        return u_a, s_a, corr_out

    @staticmethod
    def backward(ctx, grad_u, grad_s, grad_rho) -> Tuple[Tensor, Tensor, Tensor]:
        u, s, rho, u_a, s_a, chi = ctx.saved_tensors
        global mnn_core
        grad_uu, grad_us, grad_su, grad_ss, grad_chu, grad_chs = mnn_core.backward(u, s, u_a, s_a, chi)

        grad_out_u = torch.mul(grad_u, grad_uu) + torch.mul(grad_s, grad_su)
        grad_out_s = torch.mul(grad_u, grad_us) + torch.mul(grad_s, grad_ss)

        if rho.dim() != 2:
            torch.diagonal(rho, dim1=1, dim2=2).data.fill_(0.0)
        else:
            rho = rho.fill_diagonal_(0.0)

        temp_corr_grad = torch.mul(grad_rho, rho)

        if temp_corr_grad.dim() == 2:  # one sample case
            temp_corr_grad = torch.mm(chi.reshape(1, -1), temp_corr_grad)
        else:
            temp_corr_grad = torch.bmm(chi.reshape(chi.size()[0], 1, -1), temp_corr_grad)

        temp_corr_grad = 2 * temp_corr_grad.reshape(temp_corr_grad.size()[0], -1)

        corr_grad_mean = grad_chu * temp_corr_grad
        corr_grad_std = grad_chs * temp_corr_grad

        if chi.dim() == 1:
            temp_func_chi = chi.reshape(1, -1)
            chi_matrix = torch.mm(temp_func_chi.transpose(1, 0), temp_func_chi)
        else:
            temp_func_chi = chi.reshape(chi.size()[0], 1, -1)
            chi_matrix = torch.bmm(temp_func_chi.transpose(-2, -1), temp_func_chi)

        corr_grad_corr = torch.mul(chi_matrix, grad_rho)
        # set the diagonal element of corr_grad_corr to 0
        if corr_grad_corr.dim() != 2:
            torch.diagonal(corr_grad_corr, dim1=1, dim2=2).data.fill_(0.0)
        else:
            corr_grad_corr = corr_grad_corr.fill_diagonal_(0.0)

        return grad_out_u + corr_grad_mean, grad_out_s + corr_grad_std, corr_grad_corr

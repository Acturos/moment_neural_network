# -*- coding: utf-8 -*-
import math

import numpy as np
import scipy.special as scipy
import torch
from torch import Tensor

from .faddeeva_erf import FaddeevaErfi
from ..global_config import mnn_config


def chebyshev_val(x: Tensor, c: Tensor) -> Tensor:
    """
    Evaluate a Chebyshev series at points x
    x: tensor
    c: 1d tensor, a array of coefficients ordered so that the coefficients for terms of degree n are contained in c[n]
    """
    degree = c.size()[0]
    x2 = 2 * x
    c0 = c[-2]
    c1 = c[-1]
    for i in range(3, degree + 1):
        temp = c0
        c0 = c[-i] - c1
        c1 = temp + c1 * x2
    x = c0 + c1 * x
    return x


def chebyshev_val_neg(x: Tensor, c: Tensor, num_sub: int = 50, wrap: float = 4., alpha: int = 1) -> Tensor:
    delta_x = 1 / num_sub
    x = wrap / (wrap + torch.abs_(x).pow_(alpha))
    for i in range(num_sub):
        idx = torch.bitwise_and(x > delta_x * i, x <= delta_x * (i + 1))
        x[idx] = chebyshev_val(x[idx], c[i, :])
    return x


def chebyshev_val_no_transform(x: Tensor, c: Tensor, x_min: float = 0.,
                               x_max: float = 1., num_sub: int = 50) -> Tensor:
    delta_x = (x_max - x_min) / num_sub
    for i in range(num_sub):
        idx = torch.bitwise_and(torch.gt(x, x_min + delta_x * i), torch.le(x, x_min + delta_x * (i + 1)))
        x[idx] = chebyshev_val(x[idx], c[i, :])
    return x


class DawsonIntegrate(torch.nn.Module):
    """
    Args:
    """
    __constants__ = ['asym_neg_inf', 'asym_pos_inf', 'taylor', 'int_asym_neg_inf', 'div', 'deg',
                     'cheb_xmin_for_G', 'euler_gamma', 'boundary']
    div: int
    deg: int
    cheb_xmin_for_G: float
    euler_gamma: float
    boundary: float

    def __init__(self, div: int = 4, deg: int = 8, cheb_xmin_for_G: float = -6.0, boundary: float = 9.,
                 cheb_G_neg: Tensor = mnn_config.get_value('double_d1_cheb_G_neg')) -> None:
        super(DawsonIntegrate, self).__init__()
        self.div = div
        self.deg = deg
        self.euler_gamma = np.euler_gamma
        self.cheb_xmin_for_G = cheb_xmin_for_G
        self.register_buffer('cheb_G_neg', cheb_G_neg)
        self.boundary = boundary
        self.erfi = FaddeevaErfi(boundary=boundary)

    def _gpu_dawson(self, x: Tensor) -> Tensor:
        y = torch.zeros_like(x)
        region1 = torch.bitwise_or(torch.lt(x, self.cheb_xmin_for_G), torch.gt(x, - self.cheb_xmin_for_G))
        y[region1] = self.asym_neg_inf(- x[region1].abs_())

        region1.bitwise_not_()
        y[region1] = chebyshev_val_neg(- x[region1].abs_(), self.cheb_G_neg, num_sub=self.div)
        region1 = x > 0
        y[region1] = math.sqrt(math.pi) * torch.exp(x[region1].pow_(2)) - y[region1]
        return y

    @torch.jit.export
    def func_dawson(self, x: Tensor) -> Tensor:
        if x.is_cuda:
            if x.numel() > mnn_config.get_value('cpu_or_gpu'):
                return self._gpu_dawson(x)
            else:
                device = x.device
                return torch.from_numpy(scipy.erfcx(x.cpu().numpy()) * math.sqrt(math.pi) / 2).to(device=device)
        else:
            return torch.from_numpy(scipy.erfcx(x.numpy()) * math.sqrt(math.pi) / 2)

    @staticmethod
    def asym_neg_inf(x: Tensor) -> Tensor:
        y = -0.5 / x
        x2 = torch.pow(x, 2)
        output = y.clone().detach()
        for n in range(5):
            y = - y * 0.5 * (2 * n + 1) / x2
            output.add_(y)
        return output

    def integrate_asym_neg_inf(self, x: Tensor) -> Tensor:
        """
        Compute asymptotic expansion of the indefinite integral of g(x) for x<<-1
        Use recurrence relation so it only contains multiplication and addition.
        a(n+1)/a(n) = -(2n+1)/(2x^2)
        """
        temp = -0.25 * self.euler_gamma - 0.5 * torch.log(-2 * x)
        temp = temp - 1 / 8 * torch.pow(x, -2) + 3 / 32 * torch.pow(x, -4) - 5 / 32 * torch.pow(x, -6)
        return temp

    def forward(self, x: Tensor) -> Tensor:
        pos_idx = x > 0
        if x.is_cuda:
            if x.numel() < mnn_config.get_value('cpu_or_gpu'):
                device = x.device
                temp = torch.from_numpy(math.pi / 2 * scipy.erfi(x[pos_idx].cpu().numpy())).to(device=device)
            else:
                temp = math.pi / 2 * self.erfi(x[pos_idx])
        else:
            temp = torch.from_numpy(math.pi / 2 * scipy.erfi(x[pos_idx].numpy()))

        idx = torch.bitwise_or(torch.lt(x, self.cheb_xmin_for_G), torch.gt(x, - self.cheb_xmin_for_G))
        x[idx] = self.integrate_asym_neg_inf(-torch.abs_(x[idx]))
        idx.bitwise_not_()
        x[idx] = chebyshev_val_no_transform(-torch.abs_(x[idx]), self.cheb_G_neg, x_min=self.cheb_xmin_for_G,
                                            x_max=0., num_sub=self.div)

        x[pos_idx] += temp
        return x


class DoubleDawsonIntegrate(torch.nn.Module):
    """
    Args:

    """
    __constants__ = ['div', 'div_pos', 'cheb_xmas_for_H', 'boundary']

    div: int
    div_pos: int
    cheb_xmas_for_H: float
    boundary: float

    def __init__(self, div: int = 4, div_pos: int = 6, cheb_xmas_for_H: float = 4.5, boundary: float = 9.,
                 asym_neg_inf: Tensor = mnn_config.get_value('double_d2_asym_neg_inf'),
                 cheb_neg: Tensor = mnn_config.get_value('double_d2_cheb_neg'),
                 cheb_H_neg: Tensor = mnn_config.get_value('double_d2_cheb_H_neg'),
                 cheb_H_pos: Tensor = mnn_config.get_value('double_d2_cheb_H_pos')):
        super(DoubleDawsonIntegrate, self).__init__()

        self.div = div
        self.div_pos = div_pos
        self.cheb_xmas_for_H = cheb_xmas_for_H
        self.register_buffer('asym_neg_inf', asym_neg_inf)
        self.register_buffer('cheb_neg', cheb_neg)
        self.register_buffer('cheb_H_neg', cheb_H_neg)
        self.register_buffer('cheb_H_pos', cheb_H_pos)
        self.dawson1 = DawsonIntegrate(boundary=boundary)

    @torch.jit.export
    def func_dawson_2nd(self, x: Tensor) -> Tensor:
        y = torch.zeros_like(x)
        idx1 = torch.lt(x, -10.)
        y[idx1] = self.func_asym_neg_inf(x[idx1])

        idx2 = torch.gt(x, 10.)
        y[idx2] = self.func_asym_pos_inf(x[idx2])

        idx1 = torch.bitwise_not(torch.bitwise_or(idx1, idx2))
        y[idx1] = chebyshev_val_neg(-x[idx1].abs_(), self.cheb_neg, num_sub=self.div)

        idx1 = torch.bitwise_and(idx1, x > 0)
        if x.is_cuda:
            if x[idx1].numel < mnn_config.get_value('cpu_or_gpu'):
                device = x.device
                temp = torch.from_numpy(scipy.erfi(x[idx1].cpu().numpy())).to(device=device)
                y[idx1] = math.sqrt(math.pi) * torch.exp(torch.pow(x[idx1], 2)) * \
                          (0.5 * math.log(2) + 2 * self.dawson1(x[idx1]) + math.pi / 2 * temp) - y[idx1]
            else:
                y[idx1] = math.sqrt(math.pi) * torch.exp(torch.pow(x[idx1], 2)) * \
                          (0.5 * math.log(2) + 2 * self.dawson1(x[idx1]) + math.pi / 2 * self.dawson1.erfi(x[idx1])) - \
                          y[idx1]
        else:
            y[idx1] = math.sqrt(math.pi) * torch.exp(torch.pow(x[idx1], 2)) * \
                      (0.5 * math.log(2) + 2 * self.dawson1(x[idx1]) + math.pi / 2 * torch.from_numpy(
                          scipy.erfi(x[idx1].numpy()))) - y[idx1]
        return y

    def func_asym_neg_inf(self, x: Tensor, num: int = 7) -> Tensor:
        y = torch.zeros_like(x)
        for i in range(num):
            y.add_(torch.pow(x, -3 - 2 * i) * self.asym_neg_inf[i])
        return y

    def func_asym_pos_inf(self, x: Tensor) -> Tensor:
        y = math.pow(math.sqrt(math.pi) / 2, 3) * torch.exp(torch.pow(x, 2))
        if x.is_cuda:
            if x.numel() > mnn_config.get_value('cpu_or_gpu'):
                y.mul_(torch.pow(torch.erfc(-x), 2) * self.dawson1.erfi(x))
            else:
                device = y.device
                y.mul_(torch.pow(torch.erfc(-x), 2) * torch.from_numpy(scipy.erfi(x.cpu().numpy())).to(device=device))
        else:
            y.mul_(torch.pow(torch.erfc(-x), 2) * torch.from_numpy(scipy.erfi(x.numpy())))
        return y

    def func_int_asym_neg_inf(self, x: Tensor, num: int = 7) -> Tensor:
        y = torch.zeros_like(x)
        for i in range(num):
            y.add_(torch.pow(x, -2 - 2 * i) * self.asym_neg_inf[i] / (-2 - 2 * i))
        return y

    def func_int_asym_pos_inf(self, x: Tensor) -> Tensor:
        if x.is_cuda:
            if x.numel > mnn_config.get_value('cpu_or_gpu'):
                e1 = self.dawson1.erfi(x)
            else:
                device = x.device
                e1 = torch.from_numpy(scipy.erfi(x.cpu().numpy())).to(device=device)
        else:
            e1 = torch.from_numpy(scipy.erfi(x.numpy()))

        return math.pi ** 2 / 32 * (e1 - 1) * e1 * torch.pow(torch.erfc(-x), 2)

    def forward(self, x: Tensor) -> Tensor:
        idx1 = torch.lt(x, -10)
        idx2 = torch.gt(x, self.cheb_xmas_for_H)
        idx3 = torch.bitwise_and(torch.bitwise_not(idx1), x <= 0)
        idx4 = torch.bitwise_and(torch.bitwise_not(idx2), x > 0)

        x[idx1] = self.func_int_asym_neg_inf(x[idx1])
        x[idx2] = self.func_int_asym_pos_inf(x[idx2])
        x[idx3] = chebyshev_val_neg(x[idx3], self.cheb_H_neg, num_sub=self.div)
        x[idx4] = torch.exp(2 * torch.pow(x[idx4], 2)) * chebyshev_val_no_transform(x[idx4], self.cheb_H_pos,
                                                                                    x_max=self.cheb_xmas_for_H,
                                                                                    num_sub=self.div_pos)
        return x

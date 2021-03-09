# -*- coding: utf-8 -*-
import torch
from torch import Tensor



@torch.jit.script
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
    return c0 + c1 * x


def custom_erfi(x: Tensor):
    idx = x ** 2 > 720
    if torch.sum(idx) > 0:
        x[idx] = float('Inf')


class DawsonIntegrate(torch.nn.Module):
    """
    Args:
    """
    __constants__ = ['asym_neg_inf', 'asym_pos_inf', 'taylor', 'int_asym_neg_inf', 'div', 'deg',
                     'cheb_xmin_for_G', 'cheb_G_neg']
    asym_neg_inf: int
    asym_pos_inf: int
    taylor: int
    int_asym_neg_inf: int
    div: int
    deg: int
    cheb_xmin_for_G: int
    cheb_G_neg: Tensor

    def __init__(self, cheb_G_neg: Tensor = torch.load('./dawson_coefficients/d1_cheb_G_neg.tensor')):
        super(DawsonIntegrate, self).__init__()
        self.asym_neg_inf = 0
        self.asym_pos_inf = 0
        self.taylor = 0
        self.int_asym_neg_inf = 0
        self.div = 0
        self.deg = 8
        self.cheb_xmin_for_G = -6
        self.cheb_G_neg = cheb_G_neg

    @torch.jit.export
    def func_dawson(self, x: Tensor) -> Tensor:
        region1 = -torch.abs(x) < self.cheb_xmin_for_G
        region2 = ~region1
        region2_pos = x > 0
        output = torch.zeros_like(x)

        return output

    def asym_neg_inf(self, x: Tensor) -> Tensor:
        """
        Compute asymptotic expansion of the indefinite integral of g(x) for x<<-1
        Use recurrence relation so it only contains multiplication and addition.
        a(n+1)/a(n) = -(2n+1)/(2x^2)
        """


@torch.jit.script
class DoubleDawsonIntegrate:
    """
    Args:

    """
    __constants__ = ['asym_neg_inf', 'asym_pos_inf', 'taylor', 'int_asym_neg_inf', 'div', 'deg', 'div_pos',
                     'deg_pos', 'cheb_xmax_for_H', 'cheb_neg', 'cheb_H_neg', 'cheb_H_pos']
    asym_neg_inf: Tensor
    asym_pos_inf: int
    taylor: int
    int_asym_neg_inf: int
    div: int
    deg: int
    div_pos: int
    cheb_xmas_for_H: float
    cheb_neg: Tensor
    cheb_H_neg: Tensor
    cheb_H_pos: Tensor
    def __init__(self):
            pass












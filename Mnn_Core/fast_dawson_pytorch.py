# -*- coding: utf-8 -*-
import torch
from torch import Tensor
import configparser
import numpy as np
import time


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


def func_dawson(x: Tensor):
    pass


class Double_Integrate_Dawson(torch.nn.Module):
    def __init__(self):
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
        super(Double_Integrate_Dawson, self).__init__()


if __name__ == "__main__":
    NEURON = int(1e6)
    DEGREE = 50
    LOOP = 100
    inp = torch.rand(NEURON)
    coefficient = torch.rand(DEGREE)
    start = time.time()
    for _ in range(LOOP):
        _ = chebyshev_val(inp, coefficient)
    end = time.time()
    print(end - start)

    inp = torch.rand(NEURON).cuda()
    coefficient = torch.rand(DEGREE).cuda()
    start = time.time()
    for _ in range(LOOP):
        _ = chebyshev_val(inp, coefficient)
    end = time.time()
    print(end - start)








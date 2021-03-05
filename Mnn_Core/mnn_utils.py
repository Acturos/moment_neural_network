# -*- coding: utf-8 -*-
import numpy as np
import fast_dawson
import torch
from torch import Tensor
from typing import Tuple


class Param_Container:
    """
    args:
        _vol_rest: the rest voltage of a neuron
        _vol_th: the fire threshold of a neuron
        _t_ref: the refractory time of a neuoron after it fired
        _conductance: the conductance of a neuron's membrane
        _ratio: num Excitation neurons : num Inhibition neurons
        degree: from Balanced network,  the in-degree of synaptic connection follows Poisson Distribution
        with mean and variance K
    """

    def __init__(self):
        self.ratio = 0.0
        self.L = 0.05
        self.t_ref = 5.0
        self.vol_th = 20.0
        self.vol_rest = 0.0
        self.eps = 1e-5
        self.special_factor = 4 / np.sqrt(2 * np.pi * self.L) * 0.8862269251743827
        self.correction_factor = 2 / np.sqrt(2 * self.L)
        self.cut_off = 10.0
        self.ignore_t_ref = True
        self.degree = 100

    def get_degree(self):
        return self.degree

    def set_degree(self, degree):
        self.degree = degree

    def get_ratio(self):
        return self.ratio

    def set_ratio(self, ratio):
        self.ratio = ratio

    def get_t_ref(self):
        return self.t_ref

    def set_t_ref(self, t_ref):
        self.t_ref = t_ref

    def get_vol_th(self):
        return self.vol_th

    def set_vol_th(self, vol_th):
        self.vol_th = vol_th

    def get_vol_rest(self):
        return self.vol_rest

    def set_vol_rest(self, vol_rest):
        self.vol_rest = vol_rest

    def get_conductance(self):
        return self.L

    def set_conductance(self, conductance):
        self.L = conductance
        self.correction_factor = 2 / np.sqrt(2 * self.L)
        self.special_factor = 4 / np.sqrt(2 * np.pi * self.L) * 0.8862269251743827

    def get_special_factor(self):
        return self.special_factor

    def set_special_factor(self, factor):
        self.special_factor = factor

    def set_ignore_t_ref(self, flag: bool = True):
        self.ignore_t_ref = flag

    def is_ignore_t_ref(self):
        return self.ignore_t_ref

    def get_eps(self):
        return self.eps

    def set_eps(self, eps):
        self.eps = eps

    def get_cut_off(self):
        return self.cut_off

    def set_cut_off(self, cut_off):
        self.cut_off = cut_off

    def reset_params(self):
        self.ratio = 0.0
        self.L = 0.05
        self.t_ref = 5.0
        self.vol_th = 20.0
        self.vol_rest = 0.0
        self.eps = 1e-5
        self.special_factor = 4 / np.sqrt(2 * np.pi * self.L) * 0.8862269251743827
        self.correction_factor = 2 / np.sqrt(2 * self.L)
        self.cut_off = 10.0
        self.ignore_t_ref = True
        self.degree = 100

    def print_params(self):
        print("Voltage threshold:", self.get_vol_th())
        print("Voltage rest:", self.get_vol_rest())
        print("Refractory time:", self.get_t_ref())
        print("Membrane conductance:", self.get_conductance())
        print("E-I ratio:", self.get_ratio())
        print("eps: ", self.get_eps())
        print("cut_off:", self.get_cut_off())
        print("degree:", self.get_degree())


class Mnn_Core_Func(Param_Container):
    def __init__(self):
        super(Mnn_Core_Func, self).__init__()
        self.Dawson1 = fast_dawson.Dawson1()
        self.Dawson2 = fast_dawson.Dawson2()

    # compute the up and low bound of integral
    def compute_bound(self, ubar, sbar):
        indx0 = sbar > 0
        with np.errstate(all="raise"):
            ub = (self.vol_th * self.L - ubar) / (np.sqrt(self.L) * sbar + ~indx0)
            lb = (self.vol_rest * self.L - ubar) / (sbar * np.sqrt(self.L) + ~indx0)
        return ub, lb, indx0

    def forward_fast_mean(self, ubar, sbar):
        '''Calculates the mean output firing rate given the mean & std of input firing rate'''

        # Divide input domain to several regions
        indx0 = sbar > 0
        indx1 = (self.vol_th * self.L - ubar) < (self.cut_off * np.sqrt(self.L) * sbar)
        indx2 = indx0 & indx1

        mean_out = np.zeros(ubar.shape)

        # Region 0 is approx zero for sufficiently large cut_off
        # Region 1 is calculate normally
        ub = (self.vol_th * self.L - ubar[indx2]) / (sbar[indx2] * np.sqrt(self.L))
        lb = (self.vol_rest * self.L - ubar[indx2]) / (sbar[indx2] * np.sqrt(self.L))

        temp_mean = 2 / self.L * (self.Dawson1.int_fast(ub) - self.Dawson1.int_fast(lb))

        mean_out[indx2] = 1 / (temp_mean + self.t_ref)

        # Region 2 is calculated with analytical limit as sbar --> 0
        indx3 = np.logical_and(~indx0, ubar <= self.vol_th * self.L)
        indx4 = np.logical_and(~indx0, ubar > self.vol_th * self.L)
        mean_out[indx3] = 0.0
        mean_out[indx4] = 1 / (self.t_ref - 1 / self.L * np.log(1 - 1 / ubar[indx4]))

        return mean_out

    def backward_fast_mean(self, ubar, sbar, u_a):
        indx0 = sbar > 0
        indx1 = (self.vol_th * self.L - ubar) < (self.cut_off * np.sqrt(self.L) * sbar)
        indx2 = indx0 & indx1

        grad_uu = np.zeros(ubar.shape)  # Fano factor

        # Region 0 is approx zero for sufficiently large cut_off
        # Region 1 is calculate normally
        ub = (self.vol_th * self.L - ubar[indx2]) / (sbar[indx2] * np.sqrt(self.L))
        lb = (self.vol_rest * self.L - ubar[indx2]) / (sbar[indx2] * np.sqrt(self.L))

        delta_g = self.Dawson1.dawson1(ub) - self.Dawson1.dawson1(lb)
        grad_uu[indx2] = u_a[indx2] * u_a[indx2] / sbar[indx2] * delta_g * 2 / self.L / np.sqrt(self.L)

        # Region 2 is calculated with analytical limit as sbar --> 0
        indx6 = np.logical_and(~indx0, ubar <= 1)
        indx4 = np.logical_and(~indx0, ubar > 1)

        grad_uu[indx6] = 0.0
        grad_uu[indx4] = self.vol_th * u_a[indx4] * u_a[indx4] / ubar[indx4] / (ubar[indx4] - self.vol_th * self.L)

        # ---------------

        grad_us = np.zeros(ubar.shape)
        temp = self.Dawson1.dawson1(ub) * ub - self.Dawson1.dawson1(lb) * lb
        grad_us[indx2] = u_a[indx2] * u_a[indx2] / sbar[indx2] * temp * 2 / self.L

        return grad_uu, grad_us

    def forward_fast_std(self, ubar, sbar, u_a):
        '''Calculates the std of output firing rate given the mean & std of input firing rate'''

        # Divide input domain to several regions
        indx0 = sbar > 0
        indx1 = (self.vol_th * self.L - ubar) < (self.cut_off * np.sqrt(self.L) * sbar)
        indx2 = indx0 & indx1

        fano_factor = np.zeros(ubar.shape)  # Fano factor

        # Region 0 is approx zero for sufficiently large cut_off
        # Region 1 is calculate normally
        ub = (self.vol_th * self.L - ubar[indx2]) / (sbar[indx2] * np.sqrt(self.L))
        lb = (self.vol_rest * self.L - ubar[indx2]) / (sbar[indx2] * np.sqrt(self.L))

        # cached mean used
        varT = 8 / self.L / self.L * (self.Dawson2.int_fast(ub) - self.Dawson2.int_fast(lb))
        fano_factor[indx2] = varT * u_a[indx2] * u_a[indx2]

        # Region 2 is calculated with analytical limit as sbar --> 0
        fano_factor[~indx0] = (ubar[~indx0] < 1) + 0.0
        with np.errstate(invalid="raise"):
            try:
                std_out = np.sqrt(fano_factor * u_a)
            except FloatingPointError:
                print("========min batch norm input ubar & sbar ========")
                print(ubar.shape, sbar.shape)
                print(np.min(ubar), np.min(sbar), sep="\n")
                print(ubar, sbar, sep="\n")
                print("==========activate u and fano factor===============")
                print(u_a.shape, fano_factor.shape)
                print(np.min(u_a), np.min(fano_factor))
                print(u_a, fano_factor, sep="\n")
                raise FloatingPointError

        return std_out

    def backward_fast_std(self, ubar, sbar, u_a, s_a):
        '''Calculates the gradient of the std of the firing rate with respect to the mean & std of input firing rate'''

        # Divide input domain to several regions
        indx0 = sbar > 0
        indx1 = (self.vol_th * self.L - ubar) < (self.cut_off * np.sqrt(self.L) * sbar)
        indx2 = indx0 & indx1

        ub = (self.vol_th * self.L - ubar[indx2]) / (sbar[indx2] * np.sqrt(self.L))
        lb = (self.vol_rest * self.L - ubar[indx2]) / (sbar[indx2] * np.sqrt(self.L))

        grad_su = np.zeros(ubar.shape)

        delta_g = self.Dawson1.dawson1(ub) - self.Dawson1.dawson1(lb)
        delta_h = self.Dawson2.dawson2(ub) - self.Dawson2.dawson2(lb)
        delta_H = self.Dawson2.int_fast(ub) - self.Dawson2.int_fast(lb)

        temp1 = 3 / self.L / np.sqrt(self.L) * s_a[indx2] / sbar[indx2] * u_a[indx2] * delta_g
        temp2 = - 1 / 2 / np.sqrt(self.L) * s_a[indx2] / sbar[indx2] * delta_h / delta_H

        grad_su[indx2] = temp1 + temp2

        grad_ss = np.zeros(ubar.shape)

        temp_dg = self.Dawson1.dawson1(ub) * ub - self.Dawson1.dawson1(lb) * lb
        temp_dh = self.Dawson2.dawson2(ub) * ub - self.Dawson2.dawson2(lb) * lb

        grad_ss[indx2] = 3 / self.L * s_a[indx2] / sbar[indx2] * u_a[indx2] * temp_dg \
                         - 1 / 2 * s_a[indx2] / sbar[indx2] * temp_dh / delta_H

        indx4 = np.logical_and(~indx0, ubar > 1)

        grad_ss[indx4] = 1 / np.sqrt(2 * self.L) * np.power(u_a[indx4], 1.5) * np.sqrt(
            1 / (1 - ubar[indx4]) / (1 - ubar[indx4]) - 1 / ubar[indx4] / ubar[indx4])

        return grad_su, grad_ss

    def forward_fast_chi(self, ubar, sbar, u_a, s_a):
        """
        Calculates the linear response coefficient of output firing rate given the mean & std of input firing rate
        """

        # Divide input domain to several regions
        indx0 = sbar > 0
        indx1 = (self.vol_th * self.L - ubar) < (self.cut_off * np.sqrt(self.L) * sbar)
        indx2 = indx0 & indx1

        chi = np.zeros(ubar.shape)

        # Region 0 is approx zero for sufficiently large cut_off
        # Region 1 is calculate normally
        ub = (self.vol_th * self.L - ubar[indx2]) / (sbar[indx2] * np.sqrt(self.L))
        lb = (self.vol_rest * self.L - ubar[indx2]) / (sbar[indx2] * np.sqrt(self.L))

        delta_g = self.Dawson1.dawson1(ub) - self.Dawson1.dawson1(lb)
        chi[indx2] = u_a[indx2] * u_a[indx2] / s_a[indx2] * delta_g * 2 / self.L / np.sqrt(self.L)

        # delta_H = self.ds2.int_fast(ub) - self.ds2.int_fast(lb)
        # X[indx2] = np.sqrt(self.u[indx2])*delta_g/np.sqrt(delta_H)/np.sqrt(2*self.L) # alternative method

        # Region 2 is calculated with analytical limit as sbar --> 0
        indx3 = np.logical_and(~indx0, ubar <= self.vol_th * self.L)
        indx4 = np.logical_and(~indx0, ubar > self.vol_th * self.L)

        chi[indx3] = 0.0
        chi[indx4] = np.sqrt(2 / self.L) / np.sqrt(self.t_ref - 1 / self.L * np.log(1 - 1 / ubar[indx4])) / np.sqrt(
            2 * ubar[indx4] - 1)

        return chi

    def backward_fast_chi(self, ubar, sbar, u_a, chi):
        """
        Calculates the gradient of the linear response coefficient with respect to the mean & std of input firing rate
        """
        grad_uu, grad_us = self.backward_fast_mean(ubar, sbar, u_a)

        # Divide input domain to several regions
        indx0 = sbar > 0
        indx1 = (self.vol_th * self.L - ubar) < (self.cut_off * np.sqrt(self.L) * sbar)
        indx2 = indx0 & indx1

        ub = (self.vol_th * self.L - ubar[indx2]) / (sbar[indx2] * np.sqrt(self.L))
        lb = (self.vol_rest * self.L - ubar[indx2]) / (sbar[indx2] * np.sqrt(self.L))

        grad_chu = np.zeros(ubar.shape)

        tmp1 = self.Dawson1.dawson1(ub) * ub - self.Dawson1.dawson1(lb) * lb
        delta_g = self.Dawson1.dawson1(ub) - self.Dawson1.dawson1(lb)
        delta_H = self.Dawson2.int_fast(ub) - self.Dawson2.int_fast(lb)
        delta_h = self.Dawson2.dawson2(ub) - self.Dawson2.dawson2(lb)

        grad_chu[indx2] = 0.5 * chi[indx2] / u_a[indx2] * grad_uu[indx2] \
                          - np.sqrt(2) / self.L * np.sqrt(u_a[indx2] / delta_H) * tmp1 / sbar[indx2] \
                          + chi[indx2] * delta_h / delta_H / 2 / np.sqrt(self.L) / sbar[indx2]

        indx4 = np.logical_and(~indx0, ubar > 1)

        tmp_grad_uu = self.vol_th * u_a[indx4] * u_a[indx4] / ubar[indx4] / (ubar[indx4] - self.vol_th * self.L)

        grad_chu[indx4] = 1 / np.sqrt(2 * self.L) / np.sqrt(u_a[indx4] * (2 * ubar[indx4] - 1)) * tmp_grad_uu \
                          - np.sqrt(2 / self.L) / (self.vol_th * self.L) * np.sqrt(u_a[indx4]) * np.power(
            2 * ubar[indx4] - 1, -1.5)

        # -----------

        grad_chs = np.zeros(ubar.shape)

        temp_dg = 2 * self.Dawson1.dawson1(ub) * ub * ub - 2 * self.Dawson1.dawson1(lb) * lb * lb \
                  + self.vol_th * self.L / np.sqrt(self.L) / sbar[indx2]
        temp_dh = self.Dawson2.dawson2(ub) * ub - self.Dawson2.dawson2(lb) * lb
        # temp_dH = self.ds2.int_fast(ub)*ub - self.ds2.int_fast(lb)*lb

        grad_chs[indx2] = 0.5 * chi[indx2] / u_a[indx2] * grad_us[indx2] + \
                          - chi[indx2] / sbar[indx2] * (temp_dg / delta_g) \
                          + 0.5 * chi[indx2] / sbar[indx2] / delta_H * temp_dh

        return grad_chu, grad_chs


def get_cov_matrix(std_in: Tensor, corr_in: Tensor) -> Tensor:
    if std_in.dim() == 1:
        temp_std_in = std_in.view(1, -1)
        temp_std_in = torch.mm(temp_std_in.transpose(1, 0), temp_std_in)
        cov_in = torch.mul(temp_std_in, corr_in)
    else:
        temp_std_in = std_in.view(std_in.size()[0], 1, -1)
        temp_std_in = torch.bmm(temp_std_in.transpose(-2, -1), temp_std_in)
        # element-wise mul
        cov_in = torch.mul(temp_std_in, corr_in)

    return cov_in


def update_correlation(cov_out: Tensor, ext_std) -> Tuple[Tensor, Tensor]:
    if cov_out.dim() == 2:
        var_out = torch.diagonal(cov_out)
    else:
        var_out = torch.diagonal(cov_out, dim1=-2, dim2=-1)
        # Assume the external stimuli is independent with input
    if ext_std is not None:
        ext_var = torch.pow(ext_std, 2)
        var_out += ext_var
    if var_out.dim() == 1:
        temp_var_out = var_out.view(1, -1)
        temp_var_out = torch.mm(temp_var_out, torch.transpose(temp_var_out, dim0=1, dim1=0))
        # Assume every dimension of the external stimuli are mutually independent
        corr_out = torch.div(cov_out, torch.sqrt(temp_var_out))
        torch.diagonal(corr_out).data.fill_(1.0)
    else:
        temp_var_out = var_out.view(var_out.size()[0], 1, -1)
        # todo: rewrite the backward func of the following op
        temp_var_out = torch.bmm(temp_var_out, torch.transpose(temp_var_out, dim0=-2, dim1=-1))

        corr_out = torch.div(cov_out, torch.sqrt(temp_var_out))
        torch.diagonal(corr_out, dim1=-2, dim2=-1).data.fill_(1.0)
    std_out = torch.sqrt(var_out)
    return std_out, corr_out


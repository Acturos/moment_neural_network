#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2020 Yang Qi, ISTBI, Fudan University China
import matplotlib.pyplot as plt

from mnn_core.fast_dawson import *


class MomentActivation():
    def __init__(self):
        '''Moment activation function'''
        self.ds1 = Dawson1()
        self.ds2 = Dawson2()
        self.L = 0.05
        self.Tref = 5.0
        self.u = 0.0
        self.s = 0.0
        self.eps = 1e-3
        self.ub = 0.0
        self.lb = 0.0
        self.Vth = 20

    def mean(self, ubar, sbar, ignore_Tref=True, cut_off=10):
        '''Calculates the mean output firing rate given the mean & std of input firing rate'''

        # Divide input domain to several regions
        indx0 = sbar > 0
        indx1 = (self.Vth * self.L - ubar) < (cut_off * np.sqrt(self.L) * sbar)
        indx2 = indx0 & indx1

        u = np.zeros(ubar.shape)

        # Region 0 is approx zero for sufficiently large cut_off
        # Region 1 is calculate normally 
        ub = (1 - ubar[indx2]) / (sbar[indx2] * np.sqrt(self.L))
        lb = -ubar[indx2] / (sbar[indx2] * np.sqrt(self.L))

        meanT = 2 / self.L * (self.ds1.int_fast(ub) - self.ds1.int_fast(lb))

        u[indx2] = 1 / (meanT + self.Tref)

        # Region 2 is calculated with analytical limit as sbar --> 0
        indx3 = np.logical_and(~indx0, ubar <= self.Vth * self.L)
        indx4 = np.logical_and(~indx0, ubar > self.Vth * self.L)
        u[indx3] = 0.0
        u[indx4] = 1 / (self.Tref - 1 / self.L * np.log(1 - 1 / ubar[indx4]))

        # cache the results
        self.u = u

        return u

    def std(self, ubar, sbar, cut_off=10, use_cache=True):
        '''Calculates the std of output firing rate given the mean & std of input firing rate'''

        # Divide input domain to several regions
        indx0 = sbar > 0
        indx1 = (self.Vth * self.L - ubar) < (cut_off * np.sqrt(self.L) * sbar)
        indx2 = indx0 & indx1

        FF = np.zeros(ubar.shape)  # Fano factor

        # Region 0 is approx zero for sufficiently large cut_off
        # Region 1 is calculate normally 
        ub = (1 - ubar[indx2]) / (sbar[indx2] * np.sqrt(self.L))
        lb = -ubar[indx2] / (sbar[indx2] * np.sqrt(self.L))

        # cached mean used
        varT = 8 / self.L / self.L * (self.ds2.int_fast(ub) - self.ds2.int_fast(lb))
        FF[indx2] = varT * self.u[indx2] * self.u[indx2]

        # Region 2 is calculated with analytical limit as sbar --> 0
        FF[~indx0] = (ubar[~indx0] < 1) + 0.0

        s = np.sqrt(FF * self.u)

        self.s = s

        return s, FF

    def chi(self, ubar, sbar, cut_off=10, use_cache=True):
        '''Calculates the linear response coefficient of output firing rate given the mean & std of input firing rate'''

        # Divide input domain to several regions
        indx0 = sbar > 0
        indx1 = (self.Vth * self.L - ubar) < (cut_off * np.sqrt(self.L) * sbar)
        indx2 = indx0 & indx1

        X = np.zeros(ubar.shape)

        # Region 0 is approx zero for sufficiently large cut_off
        # Region 1 is calculate normally 
        ub = (1 - ubar[indx2]) / (sbar[indx2] * np.sqrt(self.L))
        lb = -ubar[indx2] / (sbar[indx2] * np.sqrt(self.L))

        delta_g = self.ds1.dawson1(ub) - self.ds1.dawson1(lb)
        X[indx2] = self.u[indx2] * self.u[indx2] / self.s[indx2] * delta_g * 2 / self.L / np.sqrt(self.L)

        # delta_H = self.ds2.int_fast(ub) - self.ds2.int_fast(lb)
        # X[indx2] = np.sqrt(self.u[indx2])*delta_g/np.sqrt(delta_H)/np.sqrt(2*self.L) # alternative method

        # Region 2 is calculated with analytical limit as sbar --> 0
        indx3 = np.logical_and(~indx0, ubar <= self.Vth * self.L)
        indx4 = np.logical_and(~indx0, ubar > self.Vth * self.L)

        X[indx3] = 0.0
        X[indx4] = np.sqrt(2 / self.L) / np.sqrt(self.Tref - 1 / self.L * np.log(1 - 1 / ubar[indx4])) / np.sqrt(
            2 * ubar[indx4] - 1)

        self.X = X
        return X

    def grad_mean(self, ubar, sbar, cut_off=10, use_cache=True):
        '''Calculates the gradient of the mean firing rate with respect to the mean & std of input firing rate'''
        # Divide input domain to several regions
        indx0 = sbar > 0
        indx1 = (self.Vth * self.L - ubar) < (cut_off * np.sqrt(self.L) * sbar)
        indx2 = indx0 & indx1

        grad_uu = np.zeros(ubar.shape)  # Fano factor

        # Region 0 is approx zero for sufficiently large cut_off
        # Region 1 is calculate normally 
        ub = (1 - ubar[indx2]) / (sbar[indx2] * np.sqrt(self.L))
        lb = -ubar[indx2] / (sbar[indx2] * np.sqrt(self.L))

        delta_g = self.ds1.dawson1(ub) - self.ds1.dawson1(lb)
        grad_uu[indx2] = self.u[indx2] * self.u[indx2] / sbar[indx2] * delta_g * 2 / self.L / np.sqrt(self.L)

        # Region 2 is calculated with analytical limit as sbar --> 0
        indx6 = np.logical_and(~indx0, ubar <= 1)
        indx4 = np.logical_and(~indx0, ubar > 1)

        grad_uu[indx6] = 0.0
        grad_uu[indx4] = self.Vth * self.u[indx4] * self.u[indx4] / ubar[indx4] / (ubar[indx4] - self.Vth * self.L)

        self.grad_uu = grad_uu

        # ---------------

        grad_us = np.zeros(ubar.shape)
        temp = self.ds1.dawson1(ub) * ub - self.ds1.dawson1(lb) * lb
        grad_us[indx2] = self.u[indx2] * self.u[indx2] / sbar[indx2] * temp * 2 / self.L

        self.grad_us = grad_us

        return grad_uu, grad_us

    def grad_std(self, ubar, sbar, cut_off=10, use_cache=True):
        '''Calculates the gradient of the std of the firing rate with respect to the mean & std of input firing rate'''

        # Divide input domain to several regions
        indx0 = sbar > 0
        indx1 = (self.Vth * self.L - ubar) < (cut_off * np.sqrt(self.L) * sbar)
        indx2 = indx0 & indx1

        ub = (1 - ubar[indx2]) / (sbar[indx2] * np.sqrt(self.L))
        lb = -ubar[indx2] / (sbar[indx2] * np.sqrt(self.L))

        grad_su = np.zeros(ubar.shape)

        delta_g = self.ds1.dawson1(ub) - self.ds1.dawson1(lb)
        delta_h = self.ds2.dawson2(ub) - self.ds2.dawson2(lb)
        delta_H = self.ds2.int_fast(ub) - self.ds2.int_fast(lb)

        temp1 = 3 / self.L / np.sqrt(self.L) * self.s[indx2] / sbar[indx2] * self.u[indx2] * delta_g
        temp2 = - 1 / 2 / np.sqrt(self.L) * self.s[indx2] / sbar[indx2] * delta_h / delta_H

        grad_su[indx2] = temp1 + temp2

        self.grad_su = grad_su

        # -----------

        grad_ss = np.zeros(ubar.shape)

        temp_dg = self.ds1.dawson1(ub) * ub - self.ds1.dawson1(lb) * lb
        temp_dh = self.ds2.dawson2(ub) * ub - self.ds2.dawson2(lb) * lb

        grad_ss[indx2] = 3 / self.L * self.s[indx2] / sbar[indx2] * self.u[indx2] * temp_dg \
                         - 1 / 2 * self.s[indx2] / sbar[indx2] * temp_dh / delta_H

        indx4 = np.logical_and(~indx0, ubar > 1)

        grad_ss[indx4] = 1 / np.sqrt(2 * self.L) * np.power(self.u[indx4], 1.5) * np.sqrt(
            1 / (1 - ubar[indx4]) / (1 - ubar[indx4]) - 1 / ubar[indx4] / ubar[indx4])

        self.grad_ss = grad_ss

        return grad_su, grad_ss

    def grad_chi(self, ubar, sbar, cut_off=10, use_cache=True):
        '''Calculates the gradient of the linear response coefficient with respect to the mean & std of input firing rate'''

        # Divide input domain to several regions
        indx0 = sbar > 0
        indx1 = (self.Vth * self.L - ubar) < (cut_off * np.sqrt(self.L) * sbar)
        indx2 = indx0 & indx1

        ub = (1 - ubar[indx2]) / (sbar[indx2] * np.sqrt(self.L))
        lb = -ubar[indx2] / (sbar[indx2] * np.sqrt(self.L))

        grad_chu = np.zeros(ubar.shape)

        tmp1 = self.ds1.dawson1(ub) * ub - self.ds1.dawson1(lb) * lb
        delta_g = self.ds1.dawson1(ub) - self.ds1.dawson1(lb)
        delta_H = self.ds2.int_fast(ub) - self.ds2.int_fast(lb)
        delta_h = self.ds2.dawson2(ub) - self.ds2.dawson2(lb)

        grad_chu[indx2] = 0.5 * self.X[indx2] / self.u[indx2] * self.grad_uu[indx2] \
                          - np.sqrt(2) / self.L * np.sqrt(self.u[indx2] / delta_H) * tmp1 / sbar[indx2] \
                          + self.X[indx2] * delta_h / delta_H / 2 / np.sqrt(self.L) / sbar[indx2]

        indx4 = np.logical_and(~indx0, ubar > 1)

        tmp_grad_uu = self.Vth * self.u[indx4] * self.u[indx4] / ubar[indx4] / (ubar[indx4] - self.Vth * self.L)

        grad_chu[indx4] = 1 / np.sqrt(2 * self.L) / np.sqrt(self.u[indx4] * (2 * ubar[indx4] - 1)) * tmp_grad_uu \
                          - np.sqrt(2 / self.L) / (self.Vth * self.L) * np.sqrt(self.u[indx4]) * np.power(
            2 * ubar[indx4] - 1, -1.5)

        self.grad_chu = grad_chu

        # -----------

        grad_chs = np.zeros(ubar.shape)

        temp_dg = 2 * self.ds1.dawson1(ub) * ub * ub - 2 * self.ds1.dawson1(lb) * lb * lb \
                  + self.Vth * self.L / np.sqrt(self.L) / sbar[indx2]
        temp_dh = self.ds2.dawson2(ub) * ub - self.ds2.dawson2(lb) * lb
        # temp_dH = self.ds2.int_fast(ub)*ub - self.ds2.int_fast(lb)*lb

        grad_chs[indx2] = 0.5 * self.X[indx2] / self.u[indx2] * self.grad_us[indx2] + \
                          - self.X[indx2] / sbar[indx2] * (temp_dg / delta_g) \
                          + 0.5 * self.X[indx2] / sbar[indx2] / delta_H * temp_dh

        self.grad_chs = grad_chs

        return grad_chu, grad_chs

    def grad_check(self):
        ''' Calculates the error of analytic v.s. numeric gradient '''

        h = 0.01

        ubar = np.arange(-1, 3, h)
        sbar = np.arange(1, 5.0, h)

        self.ubar = ubar
        self.sbar = sbar

        uv, sv = np.meshgrid(ubar, sbar)

        self.shape = uv.shape

        u = self.mean(uv.flatten(), sv.flatten())
        s, _ = self.std(uv.flatten(), sv.flatten())
        chi = self.chi(uv.flatten(), sv.flatten())

        grad_us_numeric, grad_uu_numeric = np.gradient(u.reshape(self.shape), h, h)
        grad_ss_numeric, grad_su_numeric = np.gradient(s.reshape(self.shape), h, h)
        grad_chs_numeric, grad_chu_numeric = np.gradient(chi.reshape(self.shape), h, h)

        grad_uu, grad_us = self.grad_mean(uv.flatten(), sv.flatten())
        grad_su, grad_ss = self.grad_std(uv.flatten(), sv.flatten())
        grad_chu, grad_chs = self.grad_chi(uv.flatten(), sv.flatten())

        err_grad_uu = np.abs(grad_uu - grad_uu_numeric.flatten()) / np.abs(grad_uu)
        err_grad_us = np.abs(grad_us - grad_us_numeric.flatten()) / np.abs(grad_us)
        err_grad_su = np.abs(grad_su - grad_su_numeric.flatten()) / np.abs(grad_su)
        err_grad_ss = np.abs(grad_ss - grad_ss_numeric.flatten()) / np.abs(grad_ss)
        err_grad_chu = np.abs(grad_chu - grad_chu_numeric.flatten()) / np.abs(grad_chu)
        err_grad_chs = np.abs(grad_chs - grad_chs_numeric.flatten()) / np.abs(grad_chs)

        fig = plt.figure()
        # bins = 100#
        bins = np.linspace(0, 1e-3, 101)

        ax1 = fig.add_subplot(231)
        ax1.hist(err_grad_uu, bins, density=True)
        # ax1.imshow(grad_uu.reshape(self.shape), vmin = 0, vmax = 0.05)
        ax1.set_title(r'Gradient check error $\partial\mu/\partial\bar{\mu}$')
        # ax1.set_ylabel(r'$\bar{\sigma}$')

        ax2 = fig.add_subplot(234)
        ax2.hist(err_grad_us, bins, density=True)
        # ax2.imshow(grad_uu_numeric.reshape(self.shape), vmin = 0, vmax = 0.05)
        ax2.set_title(r'Gradient check error $\partial\mu/\partial\bar{\sigma}$')

        ax3 = fig.add_subplot(232)
        ax3.hist(err_grad_su, bins, density=True)
        # ax2.imshow(grad_uu_numeric.reshape(self.shape), vmin = 0, vmax = 0.05)
        ax3.set_title(r'Gradient check error $\partial\sigma/\partial\bar{\mu}$')

        ax4 = fig.add_subplot(235)
        ax4.hist(err_grad_ss, bins, density=True)
        # ax2.imshow(grad_uu_numeric.reshape(self.shape), vmin = 0, vmax = 0.05)
        ax4.set_title(r'Gradient check error $\partial\sigma/\partial\bar{\sigma}$')

        ax5 = fig.add_subplot(233)
        ax5.hist(err_grad_chu, bins, density=True)
        # ax2.imshow(grad_uu_numeric.reshape(self.shape), vmin = 0, vmax = 0.05)
        ax5.set_title(r'Gradient check error $\partial\chi/\partial\bar{\mu}$')

        ax6 = fig.add_subplot(236)
        ax6.hist(err_grad_chs, bins, density=True)
        # ax2.imshow(grad_uu_numeric.reshape(self.shape), vmin = 0, vmax = 0.05)
        ax6.set_title(r'Gradient check error $\partial\chi/\partial\bar{\sigma}$')

        # fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.3)
        fig.tight_layout()

        return

    def visualize(self, mode='image'):
        ''' Visualize the forward and backward maps '''
        ubar = np.arange(-10, 10, 0.0501)
        sbar = np.arange(0.0, 30, 0.3)
        # ubar = np.linspace(1-0.05,1+0.05,101)
        # sbar = np.linspace(0,0.1,101)

        self.ubar = ubar
        self.sbar = sbar

        # cache

        uv, sv = np.meshgrid(ubar, sbar)

        self.shape = uv.shape

        u = self.mean(uv, sv)

        self.u = u.flatten()
        s, FF = self.std(uv.flatten(), sv.flatten())
        s = np.reshape(s, uv.shape)
        FF = np.reshape(FF, uv.shape)

        X = self.chi(uv.flatten(), sv.flatten())
        X = np.reshape(X, uv.shape)

        grad_uu, grad_us = self.grad_mean(uv.flatten(), sv.flatten())
        grad_uu = np.reshape(grad_uu, uv.shape)
        grad_us = np.reshape(grad_us, uv.shape)

        grad_su, grad_ss = self.grad_std(uv.flatten(), sv.flatten())
        grad_su = np.reshape(grad_su, uv.shape)
        grad_ss = np.reshape(grad_ss, uv.shape)

        grad_chu, grad_chs = self.grad_chi(uv.flatten(), sv.flatten())
        grad_chu = np.reshape(grad_chu, uv.shape)
        grad_chs = np.reshape(grad_chs, uv.shape)

        fig = plt.figure()

        if mode == 'image':
            ax1 = fig.add_subplot(331)
            ax1.imshow(u, extent=[ubar[0], ubar[-1], sbar[0], sbar[-1]], origin='lower',
                       aspect='auto')  # ,interpolation="lanczos"
            ax1.set_title(r'Output mean $\mu$')
            ax1.get_xaxis().set_ticks([])
            ax1.set_ylabel(r'$\bar{\sigma}$')

            ax2 = fig.add_subplot(332)
            ax2.imshow(s, extent=[ubar[0], ubar[-1], sbar[0], sbar[-1]], origin='lower',
                       aspect='auto')  # ,interpolation="lanczos"
            ax2.get_xaxis().set_ticks([])
            ax2.set_title(r'Output std $\sigma$')
            # ax2.set_ylabel(r'$\bar{\sigma}$')

            # #plot the boundary for piece-wise calculation in fast_dawson
            # Iub = lambda x, y: (1-x)/np.sqrt(self.L)/y
            # Ilb = lambda x, y: -x/np.sqrt(self.L)/y

            # xx1 = np.array([ubar[0],0])
            # ax2.plot(xx1, Ilb(xx1, 2.5), color='r',lw = 0.5)
            # xx2 = np.array([0,ubar[-1]])
            # ax2.plot(xx2, Ilb(xx2, -3), color='r',lw = 0.5)

            ax3 = fig.add_subplot(333)
            ax3.imshow(X, extent=[ubar[0], ubar[-1], sbar[0], sbar[-1]], origin='lower',
                       aspect='auto')  # ,interpolation="lanczos"
            ax3.get_xaxis().set_ticks([])
            ax3.set_title(r'Linear response $\chi$')

            ax4 = fig.add_subplot(334)
            ax4.imshow(grad_uu, extent=[ubar[0], ubar[-1], sbar[0], sbar[-1]], origin='lower', aspect='auto',
                       vmax=0.04)  # ,interpolation="lanczos")
            ax4.set_title(r'$\partial\mu/\partial\bar{\mu}$')
            ax4.set_ylabel(r'$\bar{\sigma}$')
            ax4.get_xaxis().set_ticks([])

            ax5 = fig.add_subplot(335)
            ax5.imshow(grad_su, extent=[ubar[0], ubar[-1], sbar[0], sbar[-1]], origin='lower', aspect='auto',
                       vmin=-0.05, vmax=0.15)  # ,interpolation="lanczos")
            ax5.get_xaxis().set_ticks([])
            ax5.set_title(r'$\partial\sigma/\partial\bar{\mu}$')

            ax6 = fig.add_subplot(336)
            ax6.imshow(grad_chu, extent=[ubar[0], ubar[-1], sbar[0], sbar[-1]], origin='lower', aspect='auto', vmin=-1,
                       vmax=2)  # ,interpolation="lanczos")
            ax6.get_xaxis().set_ticks([])
            ax6.set_title(r'$\partial\chi/\partial\bar{\mu}$')

            ax7 = fig.add_subplot(337)
            ax7.imshow(grad_us, extent=[ubar[0], ubar[-1], sbar[0], sbar[-1]], origin='lower', aspect='auto',
                       vmax=0.01)  # ,interpolation="lanczos")
            ax7.set_xlabel(r'$\bar{\mu}$')
            ax7.set_ylabel(r'$\bar{\sigma}$')
            ax7.set_title(r'$\partial\mu/\partial\bar{\sigma}$')

            ax8 = fig.add_subplot(338)
            ax8.imshow(grad_ss, extent=[ubar[0], ubar[-1], sbar[0], sbar[-1]], origin='lower',
                       aspect='auto')  # ,interpolation="lanczos")
            ax8.set_xlabel(r'$\bar{\mu}$')
            # ax8.set_ylabel(r'$\bar{\sigma}$')
            ax8.set_title(r'$\partial\sigma/\partial\bar{\sigma}$')

            ax9 = fig.add_subplot(339)
            ax9.imshow(grad_chs, extent=[ubar[0], ubar[-1], sbar[0], sbar[-1]], origin='lower', aspect='auto', vmin=-5,
                       vmax=10)  # ,interpolation="lanczos")
            ax9.set_xlabel(r'$\bar{\mu}$')
            ax9.set_title(r'$\partial\chi/\partial\bar{\sigma}$')
            # ax9.set_ylabel(r'$\bar{\sigma}$')


        elif mode in ['surf', 'wire']:
            ax = fig.add_subplot(331, projection='3d')
            if mode == 'surf':
                ax.plot_surface(uv, sv, u, cmap='viridis')  # ,rstride = 3,cstride=3)#, edgecolor='none')
            elif mode == 'wire':
                ax.plot_wireframe(uv, sv, u, alpha=0.7, rstride=10,
                                  cstride=80)  # ,rstride = 3,cstride=3)#, edgecolor='none')
            ax.set_xlabel(r'$\bar{\mu}$')
            ax.set_ylabel(r'$\bar{\sigma}$')
            ax.set_zlabel(r'$\mu$')
            ax.view_init(45, 45 + 180)

            ax2 = fig.add_subplot(332, projection='3d')
            if mode == 'surf':
                ax2.plot_surface(uv, sv, s, cmap='viridis')  # ,rstride = 3,cstride=3)#, edgecolor='none')
            elif mode == 'wire':
                ax2.plot_wireframe(uv, sv, s, alpha=0.7, cmap='viridis', rstride=10, cstride=80)
            ax2.set_xlabel(r'$\bar{\mu}$')
            ax2.set_ylabel(r'$\bar{\sigma}$')
            ax2.set_zlabel(r'$\sigma$')
            ax2.view_init(45, 45 + 180)

            ax3 = fig.add_subplot(333, projection='3d')
            if mode == 'surf':
                ax3.plot_surface(uv, sv, X, cmap='viridis')  # ,rstride = 3,cstride=3)#, edgecolor='none')
            elif mode == 'wire':
                ax3.plot_wireframe(uv, sv, X, alpha=0.7, cmap='viridis', rstride=10, cstride=80)
            ax3.set_xlabel(r'$\bar{\mu}$')
            ax3.set_ylabel(r'$\bar{\sigma}$')
            ax3.set_zlabel(r'$\chi$')
            ax3.view_init(45, 45 + 180)

            ax4 = fig.add_subplot(334, projection='3d')
            if mode == 'surf':
                ax4.plot_surface(uv, sv, grad_uu, cmap='viridis')  # ,rstride = 3,cstride=3)#, edgecolor='none')
            elif mode == 'wire':
                ax4.plot_wireframe(uv, sv, grad_uu, alpha=0.7, cmap='viridis', rstride=10, cstride=80)
            ax4.set_xlabel(r'$\bar{\mu}$')
            ax4.set_ylabel(r'$\bar{\sigma}$')
            ax4.zaxis.set_rotate_label(False)  # disable automatic rotation
            ax4.set_zlabel(r'$\frac{\partial\mu}{\partial\bar{\mu}}$')
            ax4.set_zlim(0, 0.04)
            ax4.view_init(45, 45 + 180)

            ax5 = fig.add_subplot(337, projection='3d')
            if mode == 'surf':
                ax5.plot_surface(uv, sv, grad_us, cmap='viridis')  # ,rstride = 3,cstride=3)#, edgecolor='none')
            elif mode == 'wire':
                ax5.plot_wireframe(uv, sv, grad_us, alpha=0.7, cmap='viridis', rstride=10, cstride=80)
            ax5.set_xlabel(r'$\bar{\mu}$')
            ax5.set_ylabel(r'$\bar{\sigma}$')
            ax5.zaxis.set_rotate_label(False)  # disable automatic rotation
            ax5.set_zlabel(r'$\frac{\partial\mu}{\partial\bar{\sigma}}$')
            ax5.set_zlim(0, 0.01)
            ax5.view_init(45, 45 + 180)

            ax6 = fig.add_subplot(335, projection='3d')
            if mode == 'surf':
                ax6.plot_surface(uv, sv, grad_su, cmap='viridis')  # ,rstride = 3,cstride=3)#, edgecolor='none')
            elif mode == 'wire':
                ax6.plot_wireframe(uv, sv, grad_su, alpha=0.7, cmap='viridis', rstride=10, cstride=80)
            ax6.set_xlabel(r'$\bar{\mu}$')
            ax6.set_ylabel(r'$\bar{\sigma}$')
            ax6.zaxis.set_rotate_label(False)  # disable automatic rotation
            ax6.set_zlabel(r'$\frac{\partial\sigma}{\partial\bar{\mu}}$')
            ax6.set_zlim(0, 0.08)
            # ax6.view_init(45, 45+180)

            ax7 = fig.add_subplot(338, projection='3d')
            if mode == 'surf':
                ax7.plot_surface(uv, sv, grad_ss, cmap='viridis')  # ,rstride = 3,cstride=3)#, edgecolor='none')
            elif mode == 'wire':
                ax7.plot_wireframe(uv, sv, grad_ss, alpha=0.7, cmap='viridis', rstride=10, cstride=80)
            ax7.set_xlabel(r'$\bar{\mu}$')
            ax7.set_ylabel(r'$\bar{\sigma}$')
            ax7.zaxis.set_rotate_label(False)  # disable automatic rotation
            ax7.set_zlabel(r'$\frac{\partial\sigma}{\partial\bar{\sigma}}$')
            ax7.set_zlim(0, 0.05)

            ax8 = fig.add_subplot(336, projection='3d')
            if mode == 'surf':
                ax8.plot_surface(uv, sv, grad_chu, cmap='viridis')  # ,rstride = 3,cstride=3)#, edgecolor='none')
            elif mode == 'wire':
                ax8.plot_wireframe(uv, sv, grad_chu, alpha=0.7, cmap='viridis', rstride=10, cstride=80)
            ax8.set_xlabel(r'$\bar{\mu}$')
            ax8.set_ylabel(r'$\bar{\sigma}$')
            ax8.zaxis.set_rotate_label(False)  # disable automatic rotation
            ax8.set_zlabel(r'$\frac{\partial\chi}{\partial\bar{\mu}}$')
            # ax8.set_zlim(-0.1,0.3)
            ax8.set_zlim(0, 0.3)
            # ax6.view_init(45, 45+180)

            ax9 = fig.add_subplot(339, projection='3d')
            if mode == 'surf':
                ax9.plot_surface(uv, sv, grad_chs, cmap='viridis')  # ,rstride = 3,cstride=3)#, edgecolor='none')
            elif mode == 'wire':
                ax9.plot_wireframe(uv, sv, grad_chs, alpha=0.7, cmap='viridis', rstride=10, cstride=80)
            ax9.set_xlabel(r'$\bar{\mu}$')
            ax9.set_ylabel(r'$\bar{\sigma}$')
            ax9.zaxis.set_rotate_label(False)  # disable automatic rotation
            ax9.set_zlabel(r'$\frac{\partial\chi}{\partial\bar{\sigma}}$')
            # ax9.set_zlim(-0.05,0.15)
            ax9.set_zlim(-0.0, 0.15)
            # ax9.view_init(45+5, 45+180)

        # ax2.set_zlim(0,10)
        plt.show()

        return


if __name__ == "__main__":
    maf = MomentActivation()
    maf.grad_check()
    maf.visualize(mode='wire')  # image, surf, wire

    # ubar = np.array([2, 3])
    # sbar = np.array([2, 5])

    # u = maf.mean(ubar,sbar)
    # print(maf.u)
    # s = maf.std(ubar,sbar)
    # print(maf.u)

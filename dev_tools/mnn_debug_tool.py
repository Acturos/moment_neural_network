# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt

from Mnn_Core.mnn_modules import *


class DebugUtils:
    @staticmethod
    def mnn_map_visualization(ubar=np.arange(-10, 10, 0.1), sbar=np.arange(0.0, 30, 0.1)):
        cmap = "rainbow"
        mnn_func = Mnn_Core_Func()
        uv, sv = np.meshgrid(ubar, sbar)
        shape = uv.shape
        uv = uv.flatten()
        sv = sv.flatten()

        u = mnn_func.forward_fast_mean(uv, sv)
        s = mnn_func.forward_fast_std(uv, sv, u)
        chi = mnn_func.forward_fast_chi(uv, sv, u, s)
        grad_uu, grad_us = mnn_func.backward_fast_mean(uv, sv, u)
        grad_su, grad_ss = mnn_func.backward_fast_std(uv, sv, u, s)
        grad_chu, grad_chs = mnn_func.backward_fast_chi(uv, sv, u, chi)

        u = u.reshape(shape)
        s = s.reshape(shape)
        chi = chi.reshape(shape)
        uv = uv.reshape(shape)
        sv = sv.reshape(shape)
        grad_uu = grad_uu.reshape(shape)
        grad_us = grad_us.reshape(shape)
        grad_su = grad_su.reshape(shape)
        grad_ss = grad_ss.reshape(shape)
        grad_chu = grad_chu.reshape(shape)
        grad_chs = grad_chs.reshape(shape)

        fig = plt.figure(figsize=(21, 21))
        fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, hspace=0.1, wspace=0.1)
        ax1 = fig.add_subplot(3, 3, 1, projection="3d")
        ax1.set_xlabel(r"$\overline{\mu}$", fontsize=16)
        ax1.set_ylabel(r"$\overline{\sigma}$", fontsize=16)
        ax1.set_zlabel(r"$\mu$", fontsize=16)
        ax1.grid(False)
        ax1.invert_xaxis()
        ax1.plot_surface(uv, sv, u, cmap=cmap)

        ax1 = fig.add_subplot(3, 3, 2, projection="3d")
        ax1.plot_surface(uv, sv, s, cmap=cmap)
        ax1.set_xlabel(r"$\overline{\mu}$", fontsize=16)
        ax1.set_ylabel(r"$\overline{\sigma}$", fontsize=16)
        ax1.set_zlabel(r"$\sigma$", fontsize=16)
        ax1.invert_xaxis()
        ax1.grid(False)

        ax1 = fig.add_subplot(3, 3, 3, projection="3d")
        ax1.plot_surface(uv, sv, chi, cmap=cmap)
        ax1.invert_xaxis()
        ax1.set_xlabel(r"$\overline{\mu}$", fontsize=16)
        ax1.set_ylabel(r"$\overline{\sigma}$", fontsize=16)
        ax1.set_zlabel(r"$\chi$", fontsize=16)
        ax1.grid(False)

        ax1 = fig.add_subplot(3, 3, 4, projection="3d")
        ax1.plot_wireframe(uv, sv, grad_uu, alpha=0.7, cmap=cmap, rstride=10, cstride=80)
        ax1.invert_xaxis()
        ax1.set_xlabel(r"$\overline{\mu}$", fontsize=16)
        ax1.set_ylabel(r"$\overline{\sigma}$", fontsize=16)
        ax1.set_zlabel(r"$\frac{\partial \mu}{\partial \overline{\mu}}$", fontsize=16)
        ax1.grid(False)
        ax1.set_zlim(-0.3, 0.3)

        ax1 = fig.add_subplot(3, 3, 5, projection="3d")
        ax1.plot_wireframe(uv, sv, grad_su, alpha=0.7, cmap=cmap, rstride=10, cstride=80)
        ax1.invert_xaxis()
        ax1.set_xlabel(r"$\overline{\mu}$", fontsize=16)
        ax1.set_ylabel(r"$\overline{\sigma}$", fontsize=16)
        ax1.set_zlabel(r"$\frac{\partial \sigma}{\partial \overline{\mu}}$", fontsize=16)
        ax1.grid(False)
        ax1.set_zlim(-0.3, 0.3)

        ax1 = fig.add_subplot(3, 3, 6, projection="3d")
        ax1.plot_wireframe(uv, sv, grad_chu, alpha=0.7, cmap=cmap, rstride=10, cstride=80)
        ax1.invert_xaxis()
        ax1.set_xlabel(r"$\overline{\mu}$", fontsize=16)
        ax1.set_ylabel(r"$\overline{\sigma}$", fontsize=16)
        ax1.set_zlabel(r"$\frac{\partial \chi}{\partial \overline{\mu}}$", fontsize=16)
        ax1.grid(False)
        ax1.set_zlim(-0.3, 0.3)

        ax1 = fig.add_subplot(3, 3, 7, projection="3d")
        ax1.plot_wireframe(uv, sv, grad_us, alpha=0.7, cmap=cmap, rstride=10, cstride=80)
        ax1.invert_xaxis()
        ax1.set_xlabel(r"$\overline{\mu}$", fontsize=16)
        ax1.set_ylabel(r"$\overline{\sigma}$", fontsize=16)
        ax1.set_zlabel(r"$\frac{\partial \mu}{\partial \overline{\sigma}}$", fontsize=16)
        ax1.grid(False)
        ax1.set_zlim(-0.1, 0.1)

        ax1 = fig.add_subplot(3, 3, 8, projection="3d")
        ax1.plot_wireframe(uv, sv, grad_ss, alpha=0.7, cmap=cmap, rstride=10, cstride=80)
        ax1.invert_xaxis()
        ax1.set_xlabel(r"$\overline{\mu}$", fontsize=16)
        ax1.set_ylabel(r"$\overline{\sigma}$", fontsize=16)
        ax1.set_zlabel(r"$\frac{\partial \sigma}{\partial \overline{\sigma}}$", fontsize=16)
        ax1.grid(False)
        ax1.set_zlim(-0.1, 0.1)

        ax1 = fig.add_subplot(3, 3, 9, projection="3d")
        ax1.plot_wireframe(uv, sv, grad_chs, alpha=0.7, cmap=cmap, rstride=10, cstride=80)
        ax1.invert_xaxis()
        ax1.set_xlabel(r"$\overline{\mu}$", fontsize=16)
        ax1.set_ylabel(r"$\overline{\sigma}$", fontsize=16)
        ax1.set_zlabel(r"$\frac{\partial \chi}{\partial \overline{\sigma}}$", fontsize=16)
        ax1.grid(False)
        ax1.set_zlim(-0.3, 0.3)

        fig.savefig("activate_map.png", dpi=300)
        plt.show()

    @staticmethod
    def batch_3d_plot(fig_size, layout, x, y, z, xlable=None, ylable=None, zlable=None, save=None, subtitle=None,
                      suptitle=None,
                      invert_x=False, invert_y=False, invert_z=False, cmap="rainbow"):
        fig = plt.figure(figsize=fig_size)
        if suptitle is not None:
            plt.suptitle(suptitle)
        if subtitle is not None:
            for i in range(layout[0] * layout[1]):
                print(i)
                ax = fig.add_subplot(layout[0], layout[1], i + 1, projection="3d")
                ax.plot_surface(x, y, z[i], cmap=cmap)
                if invert_x:
                    ax.invert_xaxis()
                if invert_y:
                    ax.invert_yaxis()
                if invert_z:
                    ax.invert_zaxis()
                if xlable is not None:
                    ax.set_xlabel(eval(xlable))
                if ylable is not None:
                    ax.set_ylabel(eval(ylable))
                if zlable is not None:
                    ax.set_zlabel(eval(zlable[i]))
                ax.set_title(eval(subtitle[i]))
        if save is not None:
            fig.savefig(save)
        plt.show()


if __name__ == '__main__':
    DebugUtils.mnn_map_visualization()

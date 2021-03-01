# -*- coding: utf-8 -*-
# Copyright 2020 Zhu Zhichao, ISTBI, Fudan University China

from Mnn_Core.mnn_pytorch import *
import itertools
from typing import Tuple


# Modules that correlation not involved:
def _ntuple(n):
    def parse(x):
        if isinstance(x, torch._six.container_abcs.Iterable):
            return x
        return tuple(itertools.repeat(x, n))

    return parse


_pair = _ntuple(2)


class Mnn_Std_Conv2d_without_Rho(torch.nn.Module):
    def __init__(self, out_channels, ext_bias=True):
        super(Mnn_Std_Conv2d_without_Rho, self).__init__()
        if ext_bias:
            self.ext_bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("ext_bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.ext_bias is not None:
            init.uniform_(self.ext_bias, 0.1, 3)

    def _std_conv_forward(self, module, inp):
        assert type(module).__name__ == "Conv2d"
        if module.padding_mode != "zeros":
            padding_repeat = tuple(x for x in reversed(module.padding) for _ in range(2))
            if self.ext_bias is not None:
                out = F.conv2d(F.pad(inp, padding_repeat, mode=module.padding_mode), torch.pow(module.weight, 2),
                               torch.pow(self.ext_bias, 2), module.stride, _pair(0), module.dilation, module.groups)
            else:
                out = F.conv2d(F.pad(inp, padding_repeat, mode=module.padding_mode), torch.pow(module.weight, 2),
                               self.ext_bias, module.stride, _pair(0), module.dilation, module.groups)

        else:
            if self.ext_bias is not None:
                out = F.conv2d(inp, torch.pow(module.weight, 2), torch.pow(self.ext_bias, 2), module.stride,
                               module.padding, module.dilation, module.groups)
            else:
                out = F.conv2d(inp, torch.pow(module.weight, 2), self.ext_bias, module.stride, module.padding,
                               module.dilation, module.groups)

        out = torch.sqrt(out)
        return out

    def forward(self, module, inp):
        return self._std_conv_forward(module, inp)


class Mnn_Dropout_without_Rho(torch.nn.Module):
    def __init__(self, p: float = 0.5, inplace: bool = False):
        super(Mnn_Dropout_without_Rho, self).__init__()
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, "
                             "but got {}".format(p))
        self.p = p
        self.inplace = inplace

    def extra_repr(self) -> str:
        return 'p={}, inplace={}'.format(self.p, self.inplace)

    def forward(self, mean, std):
        assert mean.size() == std.size()
        if self.training:
            temp = torch.zeros_like(mean)
            mean = F.dropout(mean, self.p, self.training, self.inplace)
            temp1 = torch.zeros_like(mean) + 1
            temp_map = torch.where(mean.clone().detach() != 0, temp1, temp)
            std = torch.mul(std, temp_map) / (1 - self.p)

        return mean, std


class Mnn_AvgPool2d_without_Rho(torch.nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=False,
                 count_include_pad=True, divisor_override=None):
        super(Mnn_AvgPool2d_without_Rho, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if (stride is not None) else kernel_size
        self.padding = padding
        self.count_include_pad = count_include_pad
        self.divisor_override = divisor_override
        self.ceil_mode = ceil_mode

    def extra_repr(self) -> str:
        return 'kernel_size={kernel_size}, stride={stride}, padding={padding}' \
               ', dilation={dilation}, ceil_mode={ceil_mode}'.format(**self.__dict__)

    def forward(self, mean: Tensor, std: Tensor):
        assert mean.dim() == 4
        assert mean.size() == std.size()
        mean = F.avg_pool2d(mean, self.kernel_size, self.stride, self.padding,
                            self.ceil_mode, self.count_include_pad, self.divisor_override)
        std = F.avg_pool2d(torch.pow(std, 2), self.kernel_size, self.stride, self.padding,
                           self.ceil_mode, self.count_include_pad, self.divisor_override)
        std = torch.sqrt(std / (self.kernel_size ** 2))
        return mean, std


class Mnn_AvgPool1d_without_Rho(torch.nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=False,
                 count_include_pad=True, divisor_override=None):
        super(Mnn_AvgPool1d_without_Rho, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if (stride is not None) else kernel_size
        self.padding = padding
        self.count_include_pad = count_include_pad
        self.divisor_override = divisor_override
        self.ceil_mode = ceil_mode

    def extra_repr(self) -> str:
        return 'kernel_size={kernel_size}, stride={stride}, padding={padding}' \
               ', dilation={dilation}, ceil_mode={ceil_mode}'.format(**self.__dict__)

    def forward(self, mean: Tensor, std: Tensor):
        assert mean.size() == std.size()
        mean = F.avg_pool1d(mean, self.kernel_size, self.stride, self.padding,
                            self.ceil_mode, self.count_include_pad)
        std = F.avg_pool1d(torch.pow(std, 2), self.kernel_size, self.stride, self.padding,
                           self.ceil_mode, self.count_include_pad)
        std = torch.sqrt(std / self.kernel_size)
        return mean, std


class Mnn_MaxPool2d_without_Rho(torch.nn.Module):
    def __init__(self, kernel_size: int, stride=None, padding=0, dilation=1, ceil_mode=False):
        super(Mnn_MaxPool2d_without_Rho, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if (stride is not None) else kernel_size
        self.padding = padding
        self.dilation = dilation
        self.return_indices = True
        self.ceil_mode = ceil_mode

    def extra_repr(self) -> str:
        return 'kernel_size={kernel_size}, stride={stride}, padding={padding}' \
               ', dilation={dilation}, ceil_mode={ceil_mode}'.format(**self.__dict__)

    def forward(self, mean: Tensor, std: Tensor):
        assert mean.dim() == 4
        assert mean.size() == std.size()
        mean, indices = F.max_pool2d(mean, self.kernel_size, self.stride,
                                     self.padding, self.dilation, self.ceil_mode,
                                     self.return_indices)
        N, C, H, W = indices.size()
        temp_std = torch.zeros_like(mean)
        for n in range(N):
            for c in range(C):
                for h in range(H):
                    for w in range(W):
                        p = indices[n, c, h, w].item()
                        temp_std[n, c, h, w] = std[n, c, p//std.size()[2], p % std.size()[3]]
        return mean, temp_std


class Mnn_BatchNorm1d_without_Rho(torch.nn.Module):
    def __init__(self, in_features: int, ext_std: bool = False):
        super(Mnn_BatchNorm1d_without_Rho, self).__init__()
        self.in_features = in_features
        self.bn_mean = torch.nn.BatchNorm1d(in_features)
        self.bn_mean.weight.data.fill_(2.5)
        self.bn_mean.bias.data.fill_(2.5)

        if ext_std:
            self.ext_std = Parameter(torch.Tensor(in_features))
        else:
            self.register_parameter("ext_std", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.ext_std is not None:
            init.uniform_(self.ext_std, 2, 10)

    def _compute_bn_op(self, ubar: Tensor, sbar: Tensor):
        uhat = self.bn_mean(ubar)
        if self.bn_mean.training:
            var = torch.pow(sbar, 2) * torch.pow(self.bn_mean.weight, 2) / \
                  (torch.var(ubar, dim=0, keepdim=True) + self.bn_mean.eps)
            if self.ext_std is not None:
                var += torch.pow(self.ext_std, 2)
        else:
            if self.bn_mean.track_running_stats:
                var = torch.pow(sbar, 2) * torch.pow(self.bn_mean.weight, 2) / \
                      (torch.var(ubar, dim=0, keepdim=True) + self.bn_mean.eps)
            else:
                var = torch.pow(sbar, 2) * torch.pow(self.bn_mean.weight, 2) / \
                      (torch.var(ubar, dim=0, keepdim=True) + self.bn_mean.eps)
            if self.ext_std is not None:
                var += torch.pow(self.ext_std, 2)

        shat = torch.sqrt(var)
        return uhat, shat

    def forward(self, ubar: Tensor, sbar: Tensor):
        return self._compute_bn_op(ubar, sbar)


class Mnn_Summation_Layer_without_Rho(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = False, ext_bias_std: bool = False) -> None:
        super(Mnn_Summation_Layer_without_Rho, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ratio = mnn_core_func.get_ratio()
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        if ext_bias_std:
            self.ext_bias_std = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("ext_bias_std", None)
        self.reset_parameters()

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}, ext_bias_std={}'.format(
            self.in_features, self.out_features, self.bias is not None, self.ext_bias_std is not None
        )

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / np.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)
        if self.ext_bias_std is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / np.sqrt(fan_in)
            init.uniform_(self.ext_bias_std, 0, bound)

    def forward(self, ubar: Tensor, sbar: Tensor) -> Tuple[Tensor, Tensor]:
        assert ubar.size() == sbar.size()
        ubar = F.linear(ubar, self.weight, self.bias)
        if self.ext_bias_std is not None:
            var = F.linear(torch.pow(ubar, 2), torch.pow(self.weight, 2), torch.pow(self.ext_bias_std, 2))
        else:
            var = F.linear(torch.pow(ubar, 2), torch.pow(self.weight, 2), self.ext_bias_std)
        sbar = torch.sqrt(var)
        return ubar, sbar


class Mnn_Linear_Module_without_Rho(torch.nn.Module):
    def __init__(self, input_size: int, output_size: int, bn_ext_std: bool = False):
        super(Mnn_Linear_Module_without_Rho, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.bn_ext_std = bn_ext_std

        self.linear = Mnn_Summation_Layer_without_Rho(input_size, output_size)
        self.bn = Mnn_BatchNorm1d_without_Rho(in_features=output_size, ext_std=bn_ext_std)

    def forward(self, ubar: Tensor, sbar: Tensor):
        ubar, sbar = self.linear(ubar, sbar)
        ubar, sbar = self.bn(ubar, sbar)
        u = Mnn_Activate_Mean(ubar, sbar)
        s = Mnn_Activate_Std(ubar, sbar, u)
        return u, s


# Modules that correlation is involved
class Mnn_BatchNorm1d_with_Rho(torch.nn.Module):
    def __init__(self, in_features: int, ext_std: bool = False):
        super(Mnn_BatchNorm1d_with_Rho, self).__init__()
        self.in_features = in_features
        self.bn_mean = torch.nn.BatchNorm1d(in_features)
        self.bn_mean.weight.data.fill_(2.5)
        self.bn_mean.bias.data.fill_(2.5)

        if ext_std:
            self.ext_std = Parameter(torch.Tensor(in_features))
        else:
            self.register_parameter("ext_std", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.ext_std is not None:
            init.uniform_(self.ext_std, 2, 10)

    def _compute_bn_op(self, ubar: Tensor, sbar: Tensor, corr_in: Tensor):
        uhat = self.bn_mean(ubar)
        if self.bn_mean.training:
            var = torch.pow(sbar, 2) * torch.pow(self.bn_mean.weight, 2) / \
                  (torch.var(ubar, dim=0, keepdim=True) + self.bn_mean.eps)
            if self.ext_std is not None:
                # need update correlation matrix
                std_out = torch.sqrt(var)
                cov_in = get_cov_matrix(std_out, corr_in)
                std_out, corr_in = update_correlation(cov_in, self.ext_std)
            else:
                std_out = torch.sqrt(var)
        else:
            if self.bn_mean.track_running_stats is True:
                var = torch.pow(sbar, 2) * torch.pow(self.bn_mean.weight, 2) / \
                      (self.bn_mean.running_var + self.bn_mean.eps)
            else:
                var = torch.pow(sbar, 2) * torch.pow(self.bn_mean.weight, 2) / \
                      (torch.var(ubar, dim=0, keepdim=True) + self.bn_mean.eps)
            if self.ext_std is not None:
                # need update correlation matrix
                std_out = torch.sqrt(var)
                cov_in = get_cov_matrix(std_out, corr_in)
                std_out, corr_in = update_correlation(cov_in, self.ext_std)
            else:
                std_out = torch.sqrt(var)

        return uhat, std_out, corr_in

    def forward(self, ubar: Tensor, sbar: Tensor, corr_in: Tensor):
        return self._compute_bn_op(ubar, sbar, corr_in)


class Mnn_Summation_Layer_with_Rho(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = False, ext_bias_std: bool = False) -> None:
        super(Mnn_Summation_Layer_with_Rho, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.ratio = mnn_core_func.get_ratio()
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        if ext_bias_std:
            self.ext_bias_std = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("ext_bias_std", None)
        self.reset_parameters()

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}, ext_bias_std={}'.format(
            self.in_features, self.out_features, self.bias is not None, self.ext_bias_std is not None
        )

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / np.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)
        if self.ext_bias_std is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / np.sqrt(fan_in)
            init.uniform_(self.ext_bias_std, 0, bound)

    def forward(self, mean_in: Tensor, std_in: Tensor, corr_in: Tensor):
        assert mean_in.size() == std_in.size()
        # ratio not used for std and corr
        mean_out = F.linear(mean_in, self.weight, self.bias)
        # Use corr_in and std to compute the covariance matrix
        cov_in = get_cov_matrix(std_in, corr_in)
        # cov_out = W C W^T
        cov_out = torch.matmul(self.weight, torch.matmul(cov_in, self.weight.transpose(1, 0)))
        std_out, corr_out = update_correlation(cov_out, self.ext_bias_std)

        return mean_out, std_out, corr_out


class Mnn_Linear_Module_with_Rho(torch.nn.Module):
    def __init__(self, input_size: int, output_size: int, bn_ext_std: bool = False):
        super(Mnn_Linear_Module_with_Rho, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.bn_ext_std = bn_ext_std

        self.linear = Mnn_Summation_Layer_with_Rho(input_size, output_size)
        self.bn = Mnn_BatchNorm1d_with_Rho(in_features=output_size, ext_std=bn_ext_std)

    def forward(self, u, s, rho):
        u, s, rho = self.linear(u, s, rho)

        u, s, rho = self.bn(u, s, rho)

        u_activated = Mnn_Activate_Mean.apply(u, s)
        s_activated = Mnn_Activate_Std.apply(u, s, u_activated)
        corr_activated = Mnn_Activate_Corr.apply(rho, u, s, u_activated, s_activated)

        return u_activated, s_activated, corr_activated



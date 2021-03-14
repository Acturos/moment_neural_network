# -*- coding: utf-8 -*-
from ..mnn_modules import *


class Mnn_Recurrent_Layer_with_Rho(torch.nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1, bias: bool = False,
                 bias_std: bool = False, batch_first: bool = False, bidirectional: bool = False) -> None:
        super(Mnn_Recurrent_Layer_with_Rho, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.bias_std = bias_std
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        num_directions = 2 if bidirectional else 1

        self._flat_weights_names = []
        self._all_weights = []
        for layer in range(num_layers):
            for direction in range(num_directions):
                layer_input_size = input_size if layer == 0 else hidden_size * num_directions
                w_ih = Mnn_Linear_Module_with_Rho(layer_input_size, hidden_size, bias_std)
                w_hh = Mnn_Linear_Module_with_Rho(hidden_size, hidden_size, bias_std)
                layer_params = (w_ih, w_hh)

                suffix = "_reverse" if direction == 1 else ""
                param_names = ["weight_ih_l{}{}", "weight_hh_l{}{}"]
                param_names = [x.format(layer, suffix) for x in param_names]

                for name, param in zip(param_names, layer_params):
                    setattr(self, name, param)
                self._flat_weights_names.extend(param_names)
                self._all_weights.append(param_names)

        self._flat_weights = [(lambda wn: getattr(self, wn) if hasattr(self, wn) else None)(wn)
                              for wn in self._flat_weights_names]

    def __setattr__(self, key, value):
        if hasattr(self, "_flat_weights_names") and key in self._flat_weights_names:
            idx = self._flat_weights_names.index(key)
            self.flat_weights[idx] = value
        super(Mnn_Recurrent_Layer_with_Rho, self).__setattr__(key, value)

    def reset_parameters(self) -> None:
        for name in self._flat_weights_names:
            if hasattr(self, name):
                wn = getattr(self, name)
                wn = wn.reset_parameters()
                setattr(self, name, wn)

    def forward(self, inp_u, inp_s, inp_r):
        raise NotImplementedError

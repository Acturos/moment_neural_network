# -*- coding: utf-8 -*-
import torch

_global_dict = dict()


def set_value(key, value):
    global _global_dict
    _global_dict[key] = value


def get_value(key):
    global _global_dict
    try:
        return _global_dict[key]
    
    except KeyError:
        raise Exception('No such key!')


path = __file__[:-13] + 'double_d1_cheb_G_neg.tensor'
val = torch.load(path)
set_value('double_d1_cheb_G_neg', value=val)

path = __file__[:-13] + 'double_d1_cheb_lg_neg.tensor'
val = torch.load(path)
set_value('double_d1_cheb_lg_neg', value=val)


path = __file__[:-13] + 'double_d2_asym_neg_inf.tensor'
val = torch.load(path)
set_value('double_d2_asym_neg_inf', value=val)


path = __file__[:-13] + 'double_d2_cheb_neg.tensor'
val = torch.load(path)
set_value('double_d2_cheb_neg', value=val)


path = __file__[:-13] + 'double_d2_cheb_H_neg.tensor'
val = torch.load(path)
set_value('double_d2_cheb_H_neg', value=val)


path = __file__[:-13] + 'double_d2_cheb_H_pos.tensor'
val = torch.load(path)
set_value('double_d2_cheb_H_pos', value=val)

path = __file__[:-13] + 'double_d2_cheb_H_pos.tensor'
val = torch.load(path)
set_value('double_d2_cheb_H_pos', value=val)

del path, val

set_value('cpu_or_gpu', int(1e7))

# set the parameters of LIF
set_value('t_ref', 5.0)
set_value('conductance', 0.05)
set_value('vol_th', 20.0)
set_value('vol_rest', 0.0)
set_value('cut_off', 10.0)

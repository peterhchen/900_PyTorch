import torch
import numpy as np

t = torch.Tensor ()
print('\nt:')
print(t)
print('\ntype(t):')
print(type(t))

print('\nt.dtype:')
print(t.dtype)
print('\nt.device:')
print(t.device)
print('\nt.layout:')
print(t.layout)
device = torch.device ('cuda:0')
print('\ndevice:')
print(device)
import torch
import numpy as np

t = torch.tensor ([
    [0,1,0],
    [2,0,2],
    [0,3,0]
], dtype=torch.float32)
print('\nt.sum():')
print(t.sum())
print('\nt.numel():')
print(t.numel())
print('\nt.sum().numel():')
print(t.sum().numel())
print('\nt.sum().numel() < t.numel():')
print(t.sum().numel() < t.numel())
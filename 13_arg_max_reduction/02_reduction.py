import torch
import numpy as np

t = torch.tensor ([
    [0,1,0],
    [2,0,2],
    [0,3,0]
], dtype=torch.float32)
print('\nt.sum():')
print(t.sum())
print('\nt.prod():')
print(t.prod())
print('\nt.mean():')
print(t.mean())
print('\nt.std():')
print(t.std())
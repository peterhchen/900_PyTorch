import torch
import numpy as np

t = torch.tensor ([
    [1,1,1,1],
    [2,2,2,2],
    [3,3,3,3]
], dtype=torch.float32)
print('\nt.sum(dim=0):')
print(t.sum(dim=0))
print('\nt.sum(dim=1):')
print(t.sum(dim=1))
print('\nt[0]:', t[0])
print('\nt[1]:', t[1])
print('\nt[2]:', t[2])
print('\nt[0]+t[1]+t[2]:', t[0]+t[1]+t[2])
print('\nt[0].sum():', t[0].sum())
print('\nt[1].sum():', t[1].sum())
print('\nt[2].sum():', t[2].sum())
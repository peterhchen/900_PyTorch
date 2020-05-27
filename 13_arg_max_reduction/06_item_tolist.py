import torch
import numpy as np

t = torch.tensor ([
    [1,2,3],
    [4,5,6],
    [7,8,9]
], dtype=torch.float32)
print('\nt.mean():')
print(t.mean())      
# tensor(5.)

print('\nt.mean().item():')
print(t.mean().item()) 
# 5.0

print('\nt.mean(dim=0).tolist():')
print(t.mean(dim=0).tolist())      
# [4.0, 5.0, 6.0]

print('\nt.mean(dim=0).numpy():')
print(t.mean(dim=0).numpy())   
# array([4.0, 5.0, 6.0], dtype=float32)
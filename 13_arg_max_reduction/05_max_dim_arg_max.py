import torch
import numpy as np

t = torch.tensor ([
    [1,0,0,2],
    [0,3,3,0],
    [4,0,0,5]
], dtype=torch.float32)
print('\nt.max(dim=0):')
print(t.max(dim=0))      
# tensor([4., 3., 3., 5.]), tensor([2, 1, 1, 2])

print('\nt.argmax(dim=0):')
print(t.argmax(dim=0))   
# tensor([2, 1, 1, 2]

print('\nt.max(dim=1):')
print(t.max(dim=1))      
# tensor([2., 3., 5.]), tensor([3, 1, 3])

print('\nt.argmax(dim=1):')
print(t.argmax(dim=1))   
# tensor([3, 1, 3]
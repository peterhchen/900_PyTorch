import torch
import numpy as np

t1 = torch.tensor ([
    [1,2],
    [3,4]
], dtype=torch.float32)

t2 = torch.tensor ([
    [9,8],
    [7,6]
], dtype=torch.float32)

print('\nt1 + t2:')
print(t1 + t2)

print('\nt1 + 2:')
print(t1 + 2)

print('\nt1 - 2:')
print(t1 - 2)

print('\nt1 / 2:')
print(t1 / 2)

print('\nt1.add(2):')
print(t1.add(2))

print('\nt1.sub(2):')
print(t1.sub(2))

print('\nt1.mul(2):')
print(t1.mul(2))

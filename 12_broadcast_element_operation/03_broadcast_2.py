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

print('\nnp.broadcast_to(2, t1.shape):')
print(np.broadcast_to(2, t1.shape))

print('\nt1 + 2:')
print(t1 + 2)

# npArr = np.broadcast_to(2, t1.shape)
# print('\nnpArr:')
# print(npArr)
# t3 = torch.tensor (npArr, dtype=torch.float32)
# print('\nt3:')
# print(t3)
t3 = torch.tensor (np.broadcast_to(2, t1.shape), dtype=torch.float32)
print('\nt3:')
print(t3)
print('\nt1 + t3:')
print(t1 + t3)

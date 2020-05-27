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

print('\nt1[0]:')
print(t1[0])
print('\nt1[0][0]:')
print(t1[0][0])

# Same corresponding element of t2
print('\nt2[0][0]:')
print(t2[0][0])
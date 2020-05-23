import torch
import numpy as np

data = np.array([1,2,3])

# Create with Tensor class: Tensor() [Upper case Tensor()]
t1 = torch.Tensor(data)
# Create tensor with Factory function: tensor() [Lower case tensor()]
t2 = torch.tensor(data)
# create tensor with factory function: as_tensor()
t3 = torch.as_tensor(data)
# Create tensor with factory function: from_numpy()
t4 = torch.from_numpy(data)
print('\nt1:',t1)
print('\nt2:',t2)
print('\nt3:',t3)
print('\nt3:',t4)
print('\nt1.dtype:',t1.dtype)
print('\nt2.dtype:',t2.dtype)
print('\nt3.dtype:',t3.dtype)
print('\nt3.dtype:',t4.dtype)
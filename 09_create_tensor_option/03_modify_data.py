import torch
import numpy as np

data = np.array([1,2,3])
# data[0] = 0
# data[1] = 0
# data[2] = 0
t1 = torch.Tensor(data) # Create additional copy
t2 = torch.tensor(data) # Create additional copy
t3 = torch.as_tensor(data)
t4 = torch.from_numpy(data)
data[0] = 0
data[1] = 0
data[2] = 0
print ('\nt1', t1)  # [1., 2., 3.]
print ('\nt2', t2)  # [1, 2, 3]
print ('\nt3', t3)  # [0, 0, 0]
print ('\nt4', t4)  # [0, 0, 0]
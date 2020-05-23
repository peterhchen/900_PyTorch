import torch
import numpy as np

data = np.array([1,2,3])
print ('\ntype(data):')
print (type(data))

print ('\ntorch.Tensor(data:')
print (torch.Tensor(data))
# Factor function
print('\ntorch.tensor(data):')
print(torch.tensor(data))
print('\ntorch.as_tensor(data):')
print(torch.as_tensor(data))
print('\ntorch.from_numpy(data):')
print(torch.from_numpy(data))
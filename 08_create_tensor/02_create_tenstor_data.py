import torch
import numpy as np

data = np.array([1,2,3])
print ('\ntype(data):')
print (type(data))
# Create tensor with Class: Tensor()
print ('\ntorch.Tensor(data:')
print (torch.Tensor(data))
# Create tensor Factory function: tensor()
print('\ntorch.tensor(data):')
print(torch.tensor(data))
# Create tensor with as_tensor()
print('\ntorch.as_tensor(data):')
print(torch.as_tensor(data))
# Create tensor with from_tensor()
print('\ntorch.from_numpy(data):')
print(torch.from_numpy(data))
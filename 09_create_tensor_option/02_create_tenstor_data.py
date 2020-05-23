import torch
import numpy as np

data = np.array([1,2,3])
# global data type: float32
print('\ntorch.get_default_dtype():', torch.get_default_dtype())
# tensor(float) => dtpye= float64
print('\ntorch.tensor (np.array([1.,2.,3.]):', \
    torch.tensor (np.array([1.,2.,3.])))
# explicitly data type: dtpye=torch.float64
print('\ntorch.tensor (np.array([1,2,3]), dtype=torch.float64):', \
    torch.tensor (np.array([1,2,3]), dtype=torch.float64))
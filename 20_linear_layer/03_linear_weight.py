import torch
import torch.nn as nn
in_features = torch.tensor([1,2,3,4], dtype=torch.float32)

# Example 1: We use explicitly weight matrix.
weight_matrix = torch.tensor([
    [1,2,3,4],
    [2,3,4,5],
    [3,4,5,6]
], dtype=torch.float32)
print('\nweight_matrix:')
print(weight_matrix)
print('\nweight_matrix.matmul(in_features):')
print(weight_matrix.matmul(in_features))
# tenor([30.,40.,50.])

# Example 2: PyTroch Linear Function internally 
# generate random weight matrix for us.
fc = nn.Linear (in_features=4, out_features=3)
# PyTorch module weights need to be parameters
print('\nweight_matrix2:')
print(weight_matrix)
fc.weight = nn.Parameter (weight_matrix)
# tensor ([[-0.1124, -0.997...], [...], [...]])
print ('\nfc.weight:')
print (fc.weight)
print ('\nfc(in_featues):')
print (fc(in_features))
# tensor ([29.52, 39.94, 50.45], grad_fn = <AddBacxxx>)
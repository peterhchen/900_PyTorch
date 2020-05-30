import torch
import torch.nn as nn
in_features = torch.tensor([1,2,3,4], dtype=torch.float32)
weight_matrix = torch.tensor([
    [1,2,3,4],
    [2,3,4,5],
    [3,4,5,6]
], dtype=torch.float32)
print('\nweight_matrix.matmul(in_features):')
print(weight_matrix.matmul(in_features))

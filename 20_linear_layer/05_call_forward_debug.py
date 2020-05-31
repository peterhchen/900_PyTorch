import torch
import torch.nn as nn

# Create Linear Layer
fc = nn.Linear (in_features=4, out_features=3)
# create a tensor
t = torch.tensor ([1,2,3,4], dtype=torch.float32)

# transform tenser to outout
output = fc(t)

# print output
print ('\noutput:')
print(output)
import torch

t1 = torch.tensor([
    [1,2],
    [3,4]
])
t2 = torch.tensor([
    [5,6],
    [7,8]
])

print('\ntorch.cat((t1,t2),dim=0)')
print(torch.cat((t1,t2),dim=0))
print('\ntorch.cat((t1,t2),dim=1)')
print(torch.cat((t1,t2),dim=1))
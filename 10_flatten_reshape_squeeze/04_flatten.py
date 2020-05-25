import torch

t = torch.tensor([
    [1,1,1,1],
    [2,2,2,2],
    [3,3,3,3]
])

print('\nt.reshape(1,-1)')
print(t.reshape(1,-1))
def flatten (t):
    t = t.reshape(1,-1)
    t = t.squeeze()
    return t

print('\nflatten(t)')
print(flatten(t))
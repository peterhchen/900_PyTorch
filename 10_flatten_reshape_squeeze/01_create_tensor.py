import torch

t = torch.tensor([
    [1,1,1,1],
    [2,2,2,2],
    [3,3,3,3]
])

print ('\nt.shape')
print (t.shape)
print ('\nt.Size()')
print (t.size())
print ('\nlen(t.shape)')
print (len(t.shape))
print ('\ntorch.tensor(t.shape).prod()')
print (torch.tensor(t.shape).prod())
print ('\nt.numel()')
print (t.numel())
import torch

t = torch.tensor([
    [1,1,1,1],
    [2,2,2,2],
    [3,3,3,3]
])

print ('\nt.reshape(1,12)')
print (t.reshape(1,12))
print ('\nt.reshape(1,12).squeeze()')
print (t.reshape(1,12).squeeze())
print ('\nt.reshape(1,12).squeeze().unsqueeze(dim=0)')
print (t.reshape(1,12).squeeze().unsqueeze(dim=0))
print ('\nt.reshape(1,12).squeeze().unsqueeze(dim=0).shape')
print (t.reshape(1,12).squeeze().unsqueeze(dim=0).shape)
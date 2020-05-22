import torch
t = torch.tensor([1,2,3])
print('\nt = torch.tensor([1,2,3])')
print(t)
t = t.cuda()
print('\nt.cuda():')
print(t)
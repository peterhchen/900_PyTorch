import torch

t = torch.tensor([
    [1,1,1,1],
    [2,2,2,2],
    [3,3,3,3]
])

print ('\nt.reshape(1,12)')
print (t.reshape(1,12))
print ('\nt.reshape(2,6)')
print (t.reshape(2,6))
print ('\nt.reshape(3,4)')
print (t.reshape(3,4))
print ('\nt.reshape(4,3)')
print (t.reshape(4,3))
print ('\nt.reshape(6,2)')
print (t.reshape(6,2))
print ('\nt.reshape(12,1)')
print (t.reshape(12,1))
print ('\nt.reshape(2,2,3)')
print (t.reshape(2,2,3))
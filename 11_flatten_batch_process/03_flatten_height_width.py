import torch

t1 = torch.tensor([
    [1,1,1,1],
    [1,1,1,1],
    [1,1,1,1],
    [1,1,1,1]
])

t2 = torch.tensor([
    [2,2,2,2],
    [2,2,2,2],
    [2,2,2,2],
    [2,2,2,2]
])

t3 = torch.tensor([
    [3,3,3,3],
    [3,3,3,3],
    [3,3,3,3],
    [3,3,3,3]
])

t = torch.stack((t1,t2,t3))
t = t.reshape(3,1,4,4)
print ('\nt.shape')
print(t.shape)      # torch.Size([3, 4, 4])
print('\nt')
print(t)
#batch 
#[
#   [
#       [
#           [1, 1, 1, 1],
#           [1, 1, 1, 1],
#           [1, 1, 1, 1],
#           [1, 1, 1, 1]
#       ]
#   ],
#   [
#       [
#           [2, 2, 2, 2],
#           [2, 2, 2, 2],
#           [2, 2, 2, 2],
#           [2, 2, 2, 2]
#       ]
#   ],
#   [
#       [
#           [3, 3, 3, 3],
#           [3, 3, 3, 3],
#           [3, 3, 3, 3],
#           [3, 3, 3, 3]
#       ]
#   ]
#]
t.flatten(start_dim=1).shape
print ('\nt.flatten(start_dim=1).shape:')
print (t.flatten(start_dim=1).shape)
t.flatten(start_dim=1)
print ('\nt.flatten(start_dim=1):')
print (t.flatten(start_dim=1))
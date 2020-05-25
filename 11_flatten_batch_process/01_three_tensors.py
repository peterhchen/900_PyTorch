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

print ('\nt3.shape')
print (t3.shape)    # torch.Size([4, 4])

t = torch.stack((t1,t2,t3))
print ('\nt.shape')
print(t.shape)      # torch.Size([3, 4, 4])
print('\nt')
print(t)

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
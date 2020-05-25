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
print ('\nt[0]')
print (t[0])
print ('\nt[0][0]')
print (t[0][0])
print ('\nt[0][0][0]')
print (t[0][0][0])
print ('\nt[0][0][0][0]')
print (t[0][0][0][0])

t.reshape(1,-1)[0]  # Thank you Mick!
print('\nt.reshape(1,-1)[0]:')
print(t.reshape(1,-1)[0])

t.reshape(-1)       # Thank you Aamir!
print('\nt.reshape(-1):')
print(t.reshape(-1))

t.view(t.numel())       # Thank you Ulm!
print('\nt.view(t.numel()):')
print(t.view(t.numel()))

t.flatten()       # Thank you PyTorch!
print('\nt.flatten():')
print(t.flatten())
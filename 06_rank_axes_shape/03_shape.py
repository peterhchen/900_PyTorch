import torch
# Axis
dd=[
    [1,2,3],
    [4,5,6],
    [7,8,9]
]
t = torch.tensor (dd)
print('\nt:')
print(t)
print('\ntype(t):')
print(type(t))
print('\nt.shape')
print(t.shape)

print ('\nt.reshape(1,9)')
print(t.reshape(1, 9))
print ('\nt.reshape(1,9).shape')
print(t.reshape(1, 9).shape)
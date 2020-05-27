import torch
import numpy as np

t = torch.tensor ([
    [1,0,0,2],
    [0,3,3,0],
    [4,0,0,5]
], dtype=torch.float32)
print('\nt.max():')
print(t.max())      # tensor(5.)

print('\nt.argmax():')
print(t.argmax())   # tensor(11) => What is that?

print('\nt.flatten():')
print(t.flatten())   
# tnesor([1., 0., 0., 2., 0., 3., 3., 0., 4.,0.,0.,5.])
# index = 11 is the max value
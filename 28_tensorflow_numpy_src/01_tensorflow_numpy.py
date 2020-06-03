# stack vs. concat in Python, tensorflow & numpy
# Adding a dimension
import torch
t1 = torch.tensor([1,1,1])
t1.unsqueeze(dim=0)
t1.unsqueeze(dim=1)
print(t1.shape)
print(t1.unsqueeze(dim=0).shape)
print(t1.unsqueeze(dim=1).shape)

#PyTorch: statck vs. cat
import torch
t1 = torch.tensor([1,1,1])
t2 = torch.tensor([2,2,2])
t3 = torch.tensor([3,3,3])
torch.cat(
    (t1,t2,t3)
    ,dim=0
)
torch.stack(
    (t1,t2,t3)
    ,dim=0
)
torch.cat(
    (
         t1.unsqueeze(0)
        ,t2.unsqueeze(0)
        ,t3.unsqueeze(0)
    )
    ,dim=0
)

torch.stack(
    (t1,t2,t3)
    ,dim=1
)

torch.cat(
    ( 
        t1.unsqueeze(1)
        ,t2.unsqueeze(1)
        ,t3.unsqueeze(1)
    )
    ,dim=1
)

# Stacking along the second axis
import torch
t1 = torch.tensor([1,1,1])
t2 = torch.tensor([2,2,2])
t3 = torch.tensor([3,3,3])

t1.unsqueeze(1)

t2.unsqueeze(1)

t3.unsqueeze(1)

# TensorFlow: Stack vs Concat
# pip install tensorflow==2.0.0-rc1
import tensorflow as tf

t1 = tf.constant([1,1,1])
t2 = tf.constant([2,2,2])
t3 = tf.constant([3,3,3])

tf.concat(
    (t1,t2,t3)
    ,axis=0
)

tf.stack(
    (t1,t2,t3)
    ,axis=0
)

tf.concat(
    (
         tf.expand_dims(t1, 0)
        ,tf.expand_dims(t2, 0)
        ,tf.expand_dims(t3, 0)
    )
    ,axis=0
)

tf.stack(
    (t1,t2,t3)
    ,axis=1
)

tf.concat(
    (
         tf.expand_dims(t1, 1)
        ,tf.expand_dims(t2, 1)
        ,tf.expand_dims(t3, 1)
    )    
    ,axis=1
)

# NumPy: Stack vs Concatenate
import numpy as np
t1 = np.array([1,1,1])
t2 = np.array([2,2,2])
t3 = np.array([3,3,3])
np.concatenate(
    (t1,t2,t3)
    ,axis=0
)
np.stack(
    (t1,t2,t3)
    ,axis=0
)

np.concatenate(
    (
         np.expand_dims(t1, 0)
        ,np.expand_dims(t2, 0)
        ,np.expand_dims(t3, 0)
    )
    ,axis=0
)

np.stack(
    (t1,t2,t3)
    ,axis=1
)

np.concatenate(
    (
         np.expand_dims(t1, 1)
        ,np.expand_dims(t2, 1)
        ,np.expand_dims(t3, 1)
    )
    ,axis=1
)

# Real World Example
# Joining images into a single batch
import torch
t1 = torch.zeros(3,28,28)
t2 = torch.zeros(3,28,28)
t3 = torch.zeros(3,28,28)
torch.stack(
    (t1,t2,t3)
    ,dim=0
).shape

#Joining batches into a single batch
import torch
t1 = torch.zeros(1,3,28,28)
t2 = torch.zeros(1,3,28,28)
t3 = torch.zeros(1,3,28,28)
torch.cat(
    (t1,t2,t3)
    ,dim=0
).shape

# Joining images with an existing batch
import torch
batch = torch.zeros(3,3,28,28)
t1 = torch.zeros(3,28,28)
t2 = torch.zeros(3,28,28)
t3 = torch.zeros(3,28,28)

torch.cat(
    (
        batch
        ,torch.stack(
            (t1,t2,t3)
            ,dim=0
        )
    )
    ,dim=0
).shape

import torch
batch = torch.zeros(3,3,28,28)
t1 = torch.zeros(3,28,28)
t2 = torch.zeros(3,28,28)
t3 = torch.zeros(3,28,28)

torch.cat(
    (
        batch
        ,t1.unsqueeze(0)
        ,t2.unsqueeze(0)
        ,t3.unsqueeze(0)
    )
    ,dim=0
).shape















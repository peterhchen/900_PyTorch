import torch
import torchvision
import torchvision.transforms as transforms

train_set = torchvision.datasets.FashionMNIST (
    root='./data/FashionMNIST',
    train=True,
    download=True,
    transform=transforms.Compose([transforms.ToTensor()])
)

train_loader = torch.utils.data.DataLoader (train_set, batch_size = 10)

import numpy as np
import matplotlib.pyplot as plt

torch.set_printoptions(linewidth=120)
print ('\nlen(train_set):', len(train_set))
print ('\ntrain_set.train_labels:', train_set.train_labels)
print ('\ntrain_set.train_labels.bincount():', train_set.train_labels.bincount())

sample = next(iter(train_set))
print ('\nlen(sample):', len(sample))
print ('\ntype(sample):', type(sample))
image, label = sample
print ('\nimage.shape:', image.shape)
# [1, 28, 28] 
print ('\ntype(label):', type(label))
# <class 'int'>
plt.imshow(image.squeeze(), cmap='gray')
plt.show()
print ('\nlabel:', label)

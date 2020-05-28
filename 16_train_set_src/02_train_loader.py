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

batch = next(iter(train_loader))
print ('\nlen(batch):', len(batch))
print ('\ntype(batch):', type(batch))
images, labels = batch
print ('\nimages.shape:', images.shape)
# [10, 1, 28, 28] 
print ('\nlabels.shape:', labels.shape)
# torch.Size([10])
grid = torchvision.utils.make_grid(images, nrow=10)
plt.figure(figsize=(15,15))
# display vertically
#plt.imshow(np.transpose(grid))
# display horizontally (dimension 1 and 2 exchange)
plt.imshow(np.transpose(grid, (1, 2, 0)))
plt.show()
print ('\nlabels:', labels)

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

torch.set_printoptions(linewidth=120)

train_set = torchvision.datasets.FashionMNIST(
    root='./data'
    ,train=True
    ,download=True
    ,transform=transforms.Compose([
        transforms.ToTensor()
    ])
)

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)
        
        self.fc1 = nn.Linear(in_features=12*4*4, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.out = nn.Linear(in_features=60, out_features=10)
        
    def forward(self, t):
        t = F.relu(self.conv1(t))
        t = F.max_pool2d(t, kernel_size=2, stride=2)
        
        t = F.relu(self.conv2(t))
        t = F.max_pool2d(t, kernel_size=2, stride=2)
        
        t = F.relu(self.fc1(t.reshape(-1, 12*4*4)))
        t = F.relu(self.fc2(t))
        t = self.out(t)
        
        return t

torch.set_grad_enabled(False)

# Passing a single image to the network
network = Network()
sample = next(iter(train_set))
#print('\nsample:', sample)
image, label = sample

print('\nimage.shape:', image.shape)
import matplotlib.pyplot as plt
plt.figure()
img = image.squeeze()
print('\nimg.shape:', img.shape)
plt.imshow(img) 
plt.show()  

#Neural networks usually accept batches of inputs.
#Thus, the image tensor's shape needs to be in the form 
# (batch_size × in_channels × height × width)

# This gives us a batch with size 1
print('\nimage.unsqueeze(0).shape:', image.unsqueeze(0).shape)

# image shape needs to be (batch_size × in_channels × H × W)
pred = network(image.unsqueeze(0)) 
print('\npred.shape:', pred.shape)
print('\npre:', pred)
print('\nlabel:', label)
print('\npre.argmax(dim=1)', pred.argmax(dim=1))
print('\nF.softmax(pred, dim=1):', F.softmax(pred, dim=1))
print('\nF.softmax(pred, dim=1).sum():', F.softmax(pred, dim=1).sum())

#Different instances of our network have different weights.
net1 = Network()
print('\nnet1(image.unsqueeze(0)):', net1(image.unsqueeze(0)))
net2 = Network()
print('\nnet2(image.unsqueeze(0)):', net2(image.unsqueeze(0)))
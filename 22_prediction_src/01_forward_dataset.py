import torch
import torchvision
import torchvision.transforms as transforms

train_set = torchvision.datasets.FashionMNIST (
    root='./data/FashionMNIST',
    train=True,
    download=True,
    transform=transforms.Compose([transforms.ToTensor()])
)

train_loader = torch.utils.data.DataLoader (train_set)

import torch.nn as nn
class Network (nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d (in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d (in_channels=6, out_channels=12, kernel_size=5)

        self.fc1 = nn.Linear (in_features=12*4*4, out_features=120)
        self.fc2 = nn.Linear (in_features=120, out_features=60)
        self.out = nn.Linear (in_features=60, out_features=10)

    def forward(self, t):
        # (1) input layer
        t = t
        
        # (2) hidden conv layer
        t = self.conv1(t)
        t = F.relu(t)
        t = F.max_pool2d (t, kernel_size=2, stride=2)

        # (3) hidden conv layer
        t = self.cov2(t)
        t = F.relu(t)
        t = F.max_pool2s (t, kernel_size=2, stride=2)

        # (4) hidden linear layer
        t = t.reshape(-1, 12 * 4 * 4)
        t = self.fc1(t)
        t = F.relu(t)
        
        # (5) hidden linear layer
        t = self.fc2(t)
        t = F.relu(t) 
        
        # (6) output layer
        t = self.out(t)
        #t = F.softmax(t, dim=1)
        
        return t

torch.set_grad_enabled(False)
network = Network ()
print ('\nnetwork:')
print (network)

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

torch.set_printoptions(linewidth=120)
# Check blog post on deeplizard.com for any version 
# related updates
print(torch.__version__)
print(torchvision.__version__)

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
        # (1) input layer
        t = t
        print('\n(1) input layer:t.shape:', t.shape)        
        
        # (2) hidden conv layer
        t = self.conv1(t)
        t = F.relu(t)
        print('\n(2) hidden conv1 layer:t.shape:', t.shape) 
        t = F.max_pool2d(t, kernel_size=2, stride=2)
        print('\n(2) hidden conv1 layer: F.max_pool2d: t.shape:', t.shape) 
        print('\nself.conv1.weight.shape:', self.conv1.weight.shape)
        print('\nt.min().item():', t.min().item())
        
        # (3) hidden conv layer
        t = self.conv2(t)
        t = F.relu(t)
        print('\n(3) hidden conv2 layer: t.shape:', t.shape)
        t = F.max_pool2d(t, kernel_size=2, stride=2)
        print('\n(3) hidden conv2 layer: F.max_pool2d: t.shape:', t.shape)
        print('\nself.conv2.weight.shape:', self.conv2.weight.shape)
        print('\nt.min().item():', t.min().item())
 
        # (4) hidden linear layer
        t = t.reshape(-1, 12 * 4 * 4)
        print('\n(4) hidden linear layer: reshape (): t.shape:', t.shape)
        t = self.fc1(t)
        t = F.relu(t)
        print('\n(4) hidden linear layer:t.shape:', t.shape)
        print('\nself.fc1.weight.shape:', self.fc1.weight.shape)
        print('\nt.min().item():', t.min().item())
  
        # (5) hidden linear layer
        t = self.fc2(t)
        t = F.relu(t)
        print('\n(5) hidden linear layer:t.shape:', t.shape)
        print('\nself.fc2.weight.shape:', self.fc2.weight.shape)
        print('\nt.min().item():', t.min().item())

        # (6) output layer
        t = self.out(t)
        #t = F.softmax(t, dim=1)
        print('\n(6) hidden output layer:t.shape:', t.shape)
        print('\nt.min().item():', t.min().item())

        return t

torch.set_grad_enabled(False)
network = Network()
sample = next(iter(train_set))
#print('\nsample:', sample)
image, label = sample
print('\nimage.shape:', image.shape)
print('\nlabel:', label)

network(image.unsqueeze(0))
data_loader = torch.utils.data.DataLoader(train_set, batch_size=10)
batch = next(iter(data_loader))
images, labels = batch

print('\nimages.shape:', images.shape)
print('\nlabels.shape:', labels.shape)

preds = network(images)

print('\npreds.shape:', preds.shape)
print('\npreds:', preds)
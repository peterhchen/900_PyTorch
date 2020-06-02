# Training a PyTorch CNN - Calculate Loss, Gradient & Update Weights
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

torch.set_printoptions(linewidth=120) # Display options for output
torch.set_grad_enabled(True) # Already on by default

print(torch.__version__)
print(torchvision.__version__)

def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)
        
        self.fc1 = nn.Linear(in_features=12 * 4 * 4, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.out = nn.Linear(in_features=60, out_features=10)
        
    def forward(self, t):
        # (1) input layer
        t = t
        
        # (2) hidden conv layer
        t = self.conv1(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)
        
        # (3) hidden conv layer
        t = self.conv2(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)
        
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

train_set = torchvision.datasets.FashionMNIST(
    root='./data'
    ,train=True
    ,download=True
    ,transform=transforms.Compose([
        transforms.ToTensor()
    ])
)

network = Network()

train_loader = torch.utils.data.DataLoader(train_set, batch_size=100)
batch = next(iter(train_loader))
images, labels = batch
print ('\nlabels.shape:', labels.shape)
print ('\nlabels:', labels)
# Calculating the Loss
preds = network(images)
#print('\npreds:', preds)
loss = F.cross_entropy(preds, labels) # Calculating the loss
print('\nloss:', loss)
print('\nloss.item():', loss.item())

# Calculating the Gradients
print('\nnetwork.conv1.weight.grad:',network.conv1.weight.grad)

loss.backward() # Calculating the gradients
print('\network.conv1.weight.grad.shape:', network.conv1.weight.grad.shape)
print('\nnetwork.conv1.weight.shape):', network.conv1.weight.shape)

# Updating the Weights
optimizer = optim.Adam(network.parameters(), lr=0.01)
print('\noptimizer:', optimizer)
print('\nloss.item():', loss.item())
print ('\nget_num_correct(preds, labels):', get_num_correct(preds, labels))
print ('\noptimizer.step():', optimizer.step()) # Updating the weights
preds = network(images)
loss = F.cross_entropy(preds, labels)
print('\nloss:', loss)
print('\nloss.item():', loss.item())
print ('\nget_num_correct(preds, labels):', get_num_correct(preds, labels))

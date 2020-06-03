import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torchvision.transforms as transforms

def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()

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

torch.set_grad_enabled(False)
network = Network()
train_loader = torch.utils.data.DataLoader(train_set, batch_size=1000, shuffle=True)
optimizer = optim.Adam(network.parameters(), lr=0.01)

images, labels = next(iter(train_loader))
grid = torchvision.utils.make_grid(images)

tb = SummaryWriter()
tb.add_image('images', grid)
tb.add_graph(network, images)

for epoch in range(1):
    
    total_loss = 0
    total_correct = 0
    
    for batch in train_loader: # Get Batch
        images, labels = batch 

        preds = network(images) # Pass Batch
        loss = F.cross_entropy(preds, labels) # Calculate Loss

        optimizer.zero_grad()
        # print ('loss.shape:', loss.shape)
        # print ('len(loss.shape):', len(loss.shape))
        # print ('loss:', loss)
        if (len(loss.shape) > 0):
            loss.backward() # Calculate Gradients
        optimizer.step() # Update Weights

        total_loss += loss.item()
        total_correct += get_num_correct(preds, labels)
    
    tb.add_scalar('Loss', total_loss, epoch)
    tb.add_scalar('Number Correct', total_correct, epoch)
    tb.add_scalar('Accuracy', total_correct / len(train_set), epoch)
    
    # tb.add_histogram('conv1.bias', network.conv1.bias, epoch)
    # tb.add_histogram('conv1.weight', network.conv1.weight, epoch)
    # tb.add_histogram('conv1.weight.grad', network.conv1.weight.grad, epoch)
    
    # for name, weight in network.named_parameters():
    #     print('name:', name)
    #     print('weight:', weight)
    #     print('epoch:', epoch)
    #     tb.add_histogram(name, weight, epoch)
       
    #     print("f'{name}.grad':",f'{name}.grad')
    #     print('weight.grad:', weight.grad)
    #     print('epoch:', epoch)
    #     #tb.add_histogram(f'{name}.grad', weight.grad, epoch)
    
    print("epoch", epoch, "total_correct:", total_correct, "loss:", total_loss)
    
tb.close()

# for name, weight in network.named_parameters():
#     print(name, weight.shape)

# # for name, weight in network.named_parameters():
# #     print(f'{name}.grad', weight.grad.shape)

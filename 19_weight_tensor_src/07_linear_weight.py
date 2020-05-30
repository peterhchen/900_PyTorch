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
        # Implement the forward pass
        return t

network = Network ()
print ('\nnetwork:')
print (network)
print ('\nnetwork.fc1.weight.shape:')
print (network.fc1.weight.shape)
print ('\nnetwork.fc2.weight.shape:')
print (network.fc2.weight.shape)
print ('\nnetwork.out.weight.shape:')
print (network.out.weight.shape)
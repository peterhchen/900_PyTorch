import torch.nn as nn
class Network (nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.layer = None
    
    def forward(self, t):
        t = self.layer(t)
        return t

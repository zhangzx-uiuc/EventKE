import torch.nn as nn
import torch

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.l = nn.Linear(3,4)
        self.register_parameter('bias', nn.Parameter(torch.zeros(5)))
    def forward(self):
        pass


a = Net()
print(a.bias)
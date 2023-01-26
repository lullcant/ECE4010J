import torch 
from torch import nn
class FreeEnergyBound(nn.Module):

    def __init__(self, density):
        super().__init__()

        self.density = density

    def forward(self, zk, log_jacobians):
        print(log_jacobians)
        sum_of_log_jacobians = sum(log_jacobians)
        return (-sum_of_log_jacobians - torch.log(self.density(zk))+1e-8).mean()

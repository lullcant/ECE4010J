import torch 
from torch import nn
from torch.nn import functional as F



class Planar_Flow(nn.Module):
    '''
    In the planar_flow, we need first to initiate weight , bias and coefficient which are 
    w,b,u in the paper.
    '''
    def __init__(self,dimension) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(1,dimension))
        self.bias = nn.Parameter(torch.Tensor(1))
        self.u = nn.Parameter(torch.Tensor(1,dimension))
        self.set_param()

    def set_param(self):
        self.weight.data.uniform_(-0.01,0.01)
        self.bias.data.uniform_(-0.01,0.01)
        self.u.data.uniform_(-0.01,0.01)

    def forward(self,z):
       # print(z.shape,self.weight.shape)
        F.linear(z,self.weight,self.bias)
        f_z = z + self.u*F.tanh(F.linear(z,self.weight,self.bias)) # f(z)= z + u*h(w^Tz+b) ##Question, could the activate function be changed to relu?
        return f_z
    
class Planar_Flow_Jacobian(nn.Module):
    def __init__(self,transform) -> None:
        super().__init__()
        self.weight = transform.weight
        self.bias = transform.bias
        self.u = transform.u

    def forward(self,z):
        tmp = F.linear(z,self.weight,self.bias)
        psi_z = (1-(F.tanh(tmp))**2)*self.weight
        Jacobian = 1 + torch.mm(psi_z,self.u.t())
        return torch.log(Jacobian.abs()+1e-8)


class Normalizing_Flow(nn.Module):
    '''
    In a normalizing flow, we need to get 2 things
    1. The transform itself
    2. The Jacobian
    '''
    def __init__(self,dimension,flow_length) -> None:
        super().__init__()
        self.dimension = dimension
        self.transformation = nn.Sequential(*(Planar_Flow(self.dimension) for k in range (flow_length)))
        self.Jacobian = nn.Sequential(*(Planar_Flow_Jacobian(f) for f in self.transformation))
    
    def forward(self,z):
        log_jacobian = []
        for transform,jacobian in zip(self.transformation,self.Jacobian):
            log_jacobian.append(jacobian(z))
            z=transform(z)
        zk = z
        return zk,log_jacobian
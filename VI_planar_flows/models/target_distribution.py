import numpy as np
import torch

def target_distribution(name):

    w1 = lambda z: torch.sin(2 * np.pi * z[:, 0] / 4)
    w2 = lambda z: 3 * torch.exp(-0.5 * ((z[:, 0] - 1) / 0.6) ** 2)
    w3 = lambda z: 3 * torch.sigmoid((z[:, 0] - 1) / 0.3)

    if name == "1":
        
        u = lambda z: 0.5 * ((torch.norm(z, p=2, dim=1) - 2) / 0.4)**2 - \
                 torch.log(torch.exp(-0.5*((z[:,0] - 2) / 0.6)**2) + torch.exp(-0.5*((z[:,0] + 2) / 0.6)**2) + 1e-10)
                                        
    elif name == "2":
        
        u = lambda z: 0.5 * ((z[:,1] - w1(z)) / 0.4)**2
      
    elif name == "3":

        u = lambda z: - torch.log(torch.exp(-0.5*((z[:,1] - w1(z))/0.35)**2) + torch.exp(-0.5*((z[:,1] - w1(z) + w2(z))/0.35)**2) + 1e-10)
        
    elif name == "4":

        u = lambda z: - torch.log(torch.exp(-0.5*((z[:,1] - w1(z))/0.4)**2) + torch.exp(-0.5*((z[:,1] - w1(z) + w3(z))/0.35)**2) + 1e-10)
        
    return u
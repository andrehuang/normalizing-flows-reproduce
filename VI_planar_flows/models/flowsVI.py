import torch
import torch.nn as nn
import torch.nn.functional as F

class Planar(nn.Module):   #PLanar Transfromation

    def __init__(self):

        super(Planar, self).__init__()
        self.h = nn.Tanh()
        self.w = nn.Parameter(torch.randn(1, 2).normal_(0, 0.1))
        self.b = nn.Parameter(torch.randn(1).normal_(0, 0.1))
        self.u = nn.Parameter(torch.randn(1, 2).normal_(0, 0.1))

    def forward(self, z):
        """
        Computes the following transformation:
        z' = z + u h( w^T z + b)
        Input shapes:
        shape u = (1, z_size)
        shape w = (1, z_size)
        shape b = scalar
        shape z = (batch_size, z_size).
        """
        # Equation (10)
        prod = torch.mm(z, self.w.T) + self.b
        f_z = z + self.u * self.h(prod) # this is a 3d vector
        # Equation (11)
        psi = self.w * (1 - self.h(prod) ** 2)  # w * h'(prod)
        # Equation (12)
        log_det_jacobian = torch.log(torch.abs(1 + torch.mm(psi, self.u.T)))
        
        return f_z, log_det_jacobian.T
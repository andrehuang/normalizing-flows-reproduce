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

class Coupling(nn.Module):
    def __init__(self, in_out_dim, mid_dim, hidden, mask_config):
        """Initialize a coupling layer.
        Args:
            in_out_dim: input/output dimensions.
            mid_dim: number of units in a hidden layer.
            hidden: number of hidden layers.
            mask_config: 1 if transform odd units, 0 if transform even units.
        """
        super(Coupling, self).__init__()
        self.mask_config = mask_config

        self.in_block = nn.Sequential(
            nn.Linear(in_out_dim//2, mid_dim),
            nn.ReLU())
        self.mid_block = nn.ModuleList([
            nn.Sequential(
                nn.Linear(mid_dim, mid_dim),
                nn.ReLU()) for _ in range(hidden - 1)])
        self.out_block = nn.Linear(mid_dim, in_out_dim//2)

    def forward(self, x):
        """Forward pass.
        Args:
            x: input tensor.
        Returns:
            transformed tensor.
        """
        [B, W] = list(x.size())
        x = x.reshape((B, W//2, 2))
        if self.mask_config:
            on, off = x[:, :, 0], x[:, :, 1]
        else:
            off, on = x[:, :, 0], x[:, :, 1]

        off_ = self.in_block(off)
        for i in range(len(self.mid_block)):
            off_ = self.mid_block[i](off_)
        shift = self.out_block(off_)
 
        on = on + shift

        if self.mask_config:
            x = torch.stack((on, off), dim=2)
        else:
            x = torch.stack((off, on), dim=2)
        return x.reshape((B, W))


class Scaling(nn.Module):
    """
    Log-scaling layer.
    """
    def __init__(self, dim):
        """Initialize a (log-)scaling layer.
        Args:
            dim: input/output dimensions.
        """
        super(Scaling, self).__init__()
        self.scale = nn.Parameter(
            torch.zeros((1, dim)), requires_grad=True)

    def forward(self, x):
        """Forward pass.
        Args:
            x: input tensor.
        Returns:
            transformed tensor and log-determinant of Jacobian.
        """
        log_det_J = torch.sum(self.scale)
        x = x * torch.exp(self.scale)
        return x, log_det_J



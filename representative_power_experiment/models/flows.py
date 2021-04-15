import torch
import torch.nn as nn
import torch.nn.functional as F

class Planar(nn.Module):   #PLanar Transfromation

    def __init__(self, z_size):

        super(Planar, self).__init__()
        self.h = nn.Tanh()
        self.w = nn.Parameter(torch.randn(1, z_size).normal_(0, 0.1))
        self.b = nn.Parameter(torch.randn(1).normal_(0, 0.1))
        self.u = nn.Parameter(torch.randn(1, z_size).normal_(0, 0.1))

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
        prod = torch.mm(z, self.w.T) + self.b
        # Equation (10)
        f_z = z + self.u * self.h(prod)
        # Equation (11)
        psi = self.w * (1 - self.h(prod) ** 2)  # psi = w * h'(prod)
        # Equation (12)
        log_det_jacobian = torch.log(torch.abs(1 + torch.mm(psi, self.u.T)))
        
        return f_z, log_det_jacobian.T

#First layer is a mapping from standard normal to a diagonal Gaussian, N(z|mu, var) -- mu and var are learnable parameters.
class FirstLayer(nn.Module):
    def __init__(self, z_size, learnable):
        super().__init__()
        self.mu = nn.Parameter(torch.zeros(z_size)).requires_grad_(learnable)
        self.logvar = nn.Parameter(torch.zeros(z_size)).requires_grad_(learnable)
    
    def forward(self, z):                                                 
        f_z = self.mu + self.logvar.exp() * z 
        self.sum_log_abs_det_jacobians = self.logvar.sum() #det of a diagonal matrix
        
        return f_z, self.sum_log_abs_det_jacobians * torch.ones(1,z.shape[0])



class Coupling(nn.Module):
    def __init__(self, z_size):
        """Initialize a coupling layer.
        Args:
            Coupling with only 1 hidden layer
        """
        super(Coupling, self).__init__()
        self.h = nn.Tanh()
        self.w = nn.Parameter(torch.randn(1,  z_size//2).normal_(0, 0.1))
        self.b = nn.Parameter(torch.randn(1).normal_(0, 0.1))
        self.u = nn.Parameter(torch.randn(1, z_size//2).normal_(0, 0.1))
        self.r  = torch.rand(1)

    def forward(self, z):
        """Forward pass.
        Args:
            z: input tensor.
        Returns:
            fx: transformed tensor.
        """
        [B, W] = list(z.size())
        x = z
        x = x.reshape((B, W//2, 2))
        if self.r >= 0.5:
            on, off = x[:, :, 0], x[:, :, 1]
        else:
            off, on = x[:, :, 0], x[:, :, 1]

        prod = torch.mm(off, self.w.T) + self.b
        shift = self.u * self.h(prod)

        on = on + shift # Additive coupling layer
        if self.r >= 0.5:
            fx = torch.stack((on, off), dim=1)
        else:
            fx = torch.stack((off, on), dim=1)
        fx = fx.reshape((B, W))
        return fx

class Scaling(nn.Module):
    """
    Log-scaling layer.
    """
    def __init__(self, z_size):
        """Initialize a (log-)scaling layer.
        Args:
            dim: input/output dimensions.
        """
        super(Scaling, self).__init__()
        self.scale = nn.Parameter(torch.zeros((1, z_size)), requires_grad=True)

    def forward(self, z):
        """Forward pass.
        Args:
            x: input tensor.
        Returns:
            transformed tensor and log-determinant of Jacobian.
        """
        x = z
        log_det_J = torch.sum(self.scale)
        x = x * torch.exp(self.scale)
        return x, log_det_J


class Coupling_MLP(nn.Module):
    def __init__(self, in_out_dim, mid_dim, hidden):
        """Initialize a coupling layer.
        Args:
            in_out_dim: input/output dimensions.
            mid_dim: number of units in a hidden layer.
            hidden: number of hidden layers.
        """
        super(Coupling_MLP, self).__init__()

        self.in_block = nn.Sequential(
            nn.Linear(in_out_dim//2, mid_dim),
            nn.ELU())
        self.mid_block = nn.ModuleList([
            nn.Sequential(
                nn.Linear(mid_dim, mid_dim),
                nn.ELU()) for _ in range(hidden - 1)])
        self.out_block = nn.Linear(mid_dim, in_out_dim//2)
        self.r = torch.rand(1)
        # self.r = 1.0

    def forward(self, x):
        """Forward pass.
        Args:
            x: input tensor.
        Returns:
            transformed tensor.
        """
        [B, W] = list(x.size())
        x = x.reshape((B, W//2, 2))
        # Random permutation
        # r has to be fixed for each flow
        if self.r >= 0.5:
            on, off = x[:, :, 0], x[:, :, 1]
        else:
            off, on = x[:, :, 0], x[:, :, 1]

        off_ = self.in_block(off)
        for i in range(len(self.mid_block)):
            off_ = self.mid_block[i](off_)
        shift = self.out_block(off_)
 
        on = on + shift

        if self.r >= 0.5:
            fx = torch.stack((on, off), dim=1)
        else:
            fx = torch.stack((off, on), dim=1)
        fx = fx.reshape((B, W))
        return fx

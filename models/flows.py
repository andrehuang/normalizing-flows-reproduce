import torch
import torch.nn as nn
import torch.nn.functional as F

class Planar(nn.Module):

    def __init__(self):

        super(Planar, self).__init__()
        self.h = nn.Tanh()

    def forward(self, z, u, w, b):
        """
        Computes the following transformation:
        z' = z + u h( w^T z + b)
        Input shapes:
        shape u = (batch_size, z_size, 1)
        shape w = (batch_size, 1, z_size)
        shape b = (batch_size, 1, 1)
        shape z = (batch_size, z_size).
        """

        # Equation (10)
        z = z.unsqueeze(2)
        prod = torch.bmm(w, z) + b
        f_z = z + u * self.h(prod) # this is a 3d vector
        f_z = f_z.squeeze(2) # this is a 2d vector

        # compute logdetJ
        # Equation (11)
        psi = w * (1 - self.h(prod) ** 2)  # w * h'(prod)
        # Equation (12)
        log_det_jacobian = torch.log(torch.abs(1 + torch.bmm(psi, u)))
        log_det_jacobian = log_det_jacobian.squeeze(2).squeeze(1)

        return f_z, log_det_jacobian

class Coupling(nn.Module):
    def __init__(self, in_out_dim, mid_dim, hidden):
        """Initialize a coupling layer.
        Args:
            in_out_dim: input/output dimensions.
            mid_dim: number of units in a hidden layer.
            hidden: number of hidden layers.
        """
        super(Coupling, self).__init__()

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
        # Random permutation
        perm = torch.randperm(W)
        eye = torch.eye(W)
        P = eye[perm, :].cuda()
        PT = P.t()
        x = x @ P
        x = x.reshape((B, W//2, 2))
        on, off = x[:, :, 0], x[:, :, 1]

        off_ = self.in_block(off)
        for i in range(len(self.mid_block)):
            off_ = self.mid_block[i](off_)
        shift = self.out_block(off_)
 
        on = on + shift

        x = torch.stack((on, off), dim=2)
        x = x.reshape((B, W))
        x = x @ PT
        return x

class Coupling_amor(nn.Module):
    def __init__(self):
        """Initialize a coupling layer.
        Args:
            Coupling with only 1 hidden layer
            mask_config: 1 if transform odd units, 0 if transform even units.
        """
        super(Coupling_amor, self).__init__()
        self.h = nn.Tanh()

    def forward(self, x, u, w, b):
        """Forward pass.
        Args:
            x: input tensor.
        Returns:
            transformed tensor.
        """
        [B, W] = list(x.size())
        # Random permutation
        perm = torch.randperm(W)
        eye = torch.eye(W)
        P = eye[perm, :].cuda()
        PT = P.t()
        x = x @ P

        x = x.reshape((B, W//2, 2))
        on, off = x[:, :, 0], x[:, :, 1]

        off_ = off.unsqueeze(2)
        prod = torch.bmm(w, off_) + b
        shift = u * self.h(prod)
        shift = shift.squeeze(2)

        on = on + shift # Additive coupling layer
        x = torch.stack((on, off), dim=2)

        x = x.reshape((B, W))
        x = x @ PT
        return x

class Scaling(nn.Module):
    """
    Log-scaling layer.
    """
    def __init__(self):
        """Initialize a (log-)scaling layer.
        Args:
            dim: input/output dimensions.
        """
        super(Scaling, self).__init__()
        # self.scale = nn.Parameter(
        #     torch.zeros((1, dim)), requires_grad=True)

    def forward(self, x, scale):
        """Forward pass.
        Args:
            x: input tensor.
        Returns:
            transformed tensor and log-determinant of Jacobian.
        """
        log_det_J = torch.sum(scale, dim=1)
        x = x * torch.exp(scale)
        return x, log_det_J
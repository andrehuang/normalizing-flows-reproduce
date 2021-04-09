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
        perm = torch.randperm(in_out_dim)
​        eye = torch.eye(in_out_dim)
​        self.P = eye[perm, :].cuda()
​        self.PT = self.P.t()

    def forward(self, x):
        """Forward pass.
        Args:
            x: input tensor.
        Returns:
            transformed tensor.
        """
        [B, W] = list(x.size())
        # Random permutation
        x = x @ self.P
        x = x.reshape((B, W//2, 2))
        on, off = x[:, :, 0], x[:, :, 1]

        off_ = self.in_block(off)
        for i in range(len(self.mid_block)):
            off_ = self.mid_block[i](off_)
        shift = self.out_block(off_)
 
        on = on + shift

        x = torch.stack((on, off), dim=2)
        x = x.reshape((B, W))
        x = x @ self.PT
        return x

class Coupling_amor(nn.Module):
    def __init__(self, input_dim):
        """Initialize a coupling layer.
        Args:
            Coupling with only 1 hidden layer
            mask_config: 1 if transform odd units, 0 if transform even units.
        """
        super(Coupling_amor, self).__init__()
        self.h = nn.Tanh()
        perm = torch.randperm(input_dim)
​        eye = torch.eye(input_dim)
​        self.P = eye[perm, :].cuda()
​        self.PT = self.P.t()

    def forward(self, x, u, w, b):
        """Forward pass.
        Args:
            x: input tensor.
        Returns:
            transformed tensor.
        """
        [B, W] = list(x.size())
        # Random permutation
        # perm = torch.randperm(W)
        # eye = torch.eye(W)
        # P = eye[perm, :].cuda()
        # PT = P.t()
        x = x @ self.P

        x = x.reshape((B, W//2, 2))
        on, off = x[:, :, 0], x[:, :, 1]

        off_ = off.unsqueeze(2)
        prod = torch.bmm(w, off_) + b
        shift = u * self.h(prod)
        shift = shift.squeeze(2)

        on = on + shift # Additive coupling layer
        x = torch.stack((on, off), dim=2)

        x = x.reshape((B, W))
        x = x @ self.PT
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
    
class Sylvester(nn.Module):
    """
    Sylvester normalizing flow.
    """

    def __init__(self, M):

        super(Sylvester, self).__init__()
        self.h = nn.Tanh()

    def forward(self, z, Q, R, R_tilde, b):
        """
        Computes the transformation of Equation (13):
        z' = z + QR h(R_tilde Q^T z + b)
        Input shapes:
        shape z = (batch_size, z_size)
        shape R = (batch_size, M, M)
        shape R_tilde = (batch_size, M, M)
        shape Q = (batch_size, z_size , M)
        shape b  = (batch_size, M)
        """

        ##Computations for Equation (13)
        z = z.unsqueeze(2) 
        b = b.unsqueeze(2)         
        RQ = torch.bmm(R_tilde, Q.transpose(2, 1))
        prod  = torch.bmm(RQ, z) + b
        QR = torch.bmm(Q, R)
        
        #Equation (13)
        f_z = z + torch.bmm(QR, self.h(prod))
        f_z = f_z.squeeze(2)
        
        ##Computations for Equation (14)
        R_diag = torch.diagonal(R, dim1=1, dim2=2)
        R_tidle_diag = torch.diagonal(R_tilde, dim1=1, dim2=2)

        RR_diag = R_diag * R_tidle_diag                     #RR_diag.shape = [batch_size, M]   
        h_der = (1 - self.h(prod) ** 2).squeeze(2)          #h'(R_tidle Q^T z + b)
        
        #diagonal of the argument of det in Equation (14)
        det_J_diag = 1 + h_der * RR_diag 
        
        log_det_J_diag = det_J_diag.abs().log()                     
        log_det_jacobian = log_det_J_diag.sum(-1) #det of diagonal matrix

        return f_z, log_det_jacobian
    
class AffineCoupling(torch.nn.Module):
    """
    Input:
    - input_output_dim, mid_dim, hidden_dim=1
    Output:
    - Transformed x->z
    - Log-determinant of Jacobian
    """
    
    
    def __init__(self, input_output_dim, mid_dim, hidden_dim=1):
        super(AffineCoupling, self).__init__()
        self.input_output_dim = input_output_dim
        self.mid_dim = mid_dim
        self.hidden_dim = hidden_dim
        self.s = nn.Sequential(nn.Linear(input_output_dim//2, mid_dim), nn.ReLU(), nn.Linear(mid_dim, mid_dim), nn.ReLU(), nn.Linear(mid_dim, input_output_dim//2))
        self.t = nn.Sequential(nn.Linear(input_output_dim//2, mid_dim), nn.ReLU(), nn.Linear(mid_dim, mid_dim), nn.ReLU(), nn.Linear(mid_dim, input_output_dim//2))
        # self.perm = torch.randperm(self.input_output_dim)
        # self.eye = torch.eye(self.perm)
        # self.P = self.eye[self.perm, :]
        # self.P_t = self.P.t()
        perm = torch.randperm(self.input_output_dim)
        eye = torch.eye(self.input_output_dim)
        self.P = eye[perm, :]
        self.PT = self.P.t()

    def forward(self, x):
        d = self.input_output_dim//2
        x = x @ self.P
        x1, x2 = x[:, :d], x[:, d:]
        scale = self.s(x1)
        translate = self.t(x1)
        z1 = x1
        z2 = x2 * torch.exp(scale)
        z3 = translate
        z4 = z2 + z3
        z = torch.cat((z1, z4), dim=1)
        z = z @ self.PT
        log_det_j = scale.sum(-1)
        
        return z, log_det_j

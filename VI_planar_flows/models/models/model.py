import torch
import torch.nn as nn
from models.flows import Planar, FirstLayer
from models.flows import Coupling, Scaling

class PlanarFlow(nn.Module):  ##PlanarVI without VAEs
    """
    Stacking planar transformations
    """

    def __init__(self, K, z_size, learnable_affine):
        super(PlanarFlow, self).__init__()

        # Initialize log-det-jacobian to zero
        self.log_det_j = 0.
        # Flow parameters
        flow = Planar
        q_0 = FirstLayer
        self.num_flows = K
        
        #First layer 
        flow_0 = q_0(z_size, learnable_affine)
        self.add_module('flow_' + str(0), flow_0)
        
        # Normalizing flow layers
        for k in range(1,self.num_flows+1):
            flow_k = flow(z_size)
            self.add_module('flow_' + str(k), flow_k)                                                              


    def forward(self, z):
        """
        Forward pass with planar flows for the transformation z_0 -> z_1 -> ... -> z_k.
        Log determinant is computed as log_det_j = N E_q_z0[\sum_k log |det dz_k/dz_k-1| ].
        """

        self.log_det_j = 0.
        
        # Normalizing flows            
        for k in range(self.num_flows+1):           
            flow_k = getattr(self, 'flow_' + str(k))
            z, log_det_jacobian = flow_k(z)                                                                      
            self.log_det_j += log_det_jacobian


        return z, self.log_det_j 
    
class NICEFlow(nn.Module):
    def __init__(self, K, 
        in_out_dim, mid_dim, hidden=1, mask_config=0):
        """Initialize a NICE flow.
        Args:
            K: number of coupling layers.
            in_out_dim: input/output dimensions.
            mid_dim: number of units in a hidden layer.
            hidden: number of hidden layers.
            mask_config: 1 if transform odd units, 0 if transform even units.
        """
        super(NICEFlow, self).__init__()
        self.in_out_dim = in_out_dim

        self.coupling = nn.ModuleList([
            Coupling(in_out_dim=in_out_dim, 
                     mid_dim=mid_dim, 
                     hidden=hidden, 
                     mask_config=(mask_config+i)%2) \
            for i in range(K)])
        self.scaling = Scaling(in_out_dim)

    def forward(self, x):
        """Transformation f: X -> Z (inverse of g).
        Args:
            x: tensor in data space X.
        Returns:
            transformed tensor in latent space Z.
        """
        # Add random permutation before coupling as in Normalizing flow paper
        random_perm = torch.randperm(x.shape[1])
        x = x[:, random_perm]
        for i in range(len(self.coupling)):
            x = self.coupling[i](x)
        return self.scaling(x)
    
    
    
class RealNVPFlow(nn.Module):
    def __init__(self, num_flows, input_output_dim, mid_dim, hidden=1):
        """Initialize a RealNV?P flow.
        Args:
            num_flows: number of coupling layers.
            input_output_dim: input/output dimensions.
            mid_dim: number of units in a hidden layer.
            hidden: number of hidden layers.
        """
        super(RealNVPFlow, self).__init__()
        self.num_flows = num_flows
        self.input_output_dim = input_output_dim
        self.mid_dim = mid_dim
        self.hidden = hidden
        self.log_det_j = 0

        for k in range(self.num_flows):
            mask = 0 if k%2==0 else 1
            flow_k = AffineCoupling(self.input_output_dim, self.mid_dim, self.hidden, mask)
            self.add_module('flow_' + str(k), flow_k)

    def forward(self, x):
        """Transformation f: X -> Z (inverse of g).
        Args:
            x: tensor in data space X.
        Returns:
            transformed tensor in latent space Z.
        """
        for k in range(self.num_flows):
            flow_k = getattr(self, 'flow_' + str(k))
            x, logdet = flow_k(x)
            self.log_det_j += logdet

        return x,  self.log_det_j

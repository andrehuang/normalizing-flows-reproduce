import torch
import torch.nn as nn
import torch.nn.functional as F
from models.flows import Planar
from models.flows import Coupling, Scaling
# Model function for decoder and encoder

# A simple MLP architecture
class MLP(nn.Module):
    def __init__(
          self,
          input_dim,
          features,
          depth,
          num_outputs=None
    ):
        super().__init__()

        self.num_outputs = num_outputs
        self.input_dim = input_dim
        self.first = nn.Linear(input_dim, features)
        self.layers = nn.ModuleList(
            [nn.Linear(features, features) for i in range(depth)]
        )
        if num_outputs is not None:
            self.last = nn.Linear(features, num_outputs)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        x = self.first(x)

        for layer in self.layers:
            x = F.relu(x)
            x = layer(x)

        if self.num_outputs is not None:
            x = self.last(x)

        return x

def MLP_encoder(args):
    model = MLP(input_dim=784, features=400, depth=1, num_outputs=args.encoder_dim)
    return model

def MLP_decoder(args):
    model = MLP(input_dim=args.z_size, features=400, depth=1, num_outputs=args.decoder_dim)
    return model



class PlanarFlow(nn.Module):  ##PlanarVI without VAEs
    """
    Stacking planar transformations
    """

    def __init__(self, K: int = 6):
        super(PlanarFlow, self).__init__()

        # Initialize log-det-jacobian to zero
        self.log_det_j = 0.
        # Flow parameters
        flow = Planar
        self.num_flows = K

        self.w = nn.Parameter(torch.randn(K, 1, 2).normal_(0, 0.1))
        self.b = nn.Parameter(torch.randn(K, 1).normal_(0, 0.1))
        self.u = nn.Parameter(torch.randn(K, 1, 2).normal_(0, 0.1))

           
        # Normalizing flow layers
        for k in range(self.num_flows):
            flow_k = flow()
            self.add_module('flow_' + str(k), flow_k)                                                              


    def forward(self, z):
        """
        Forward pass with planar flows for the transformation z_0 -> z_1 -> ... -> z_k.
        Log determinant is computed as log_det_j = N E_q_z0[\sum_k log |det dz_k/dz_k-1| ].
        """

        self.log_det_j = 0.
        
        # Normalizing flows            
        for k in range(self.num_flows):           
            flow_k = getattr(self, 'flow_' + str(k))
            z, log_det_jacobian = flow_k(z, u[k], w[k], b[k])                                                                        
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
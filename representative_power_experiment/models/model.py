import torch
import torch.nn as nn
from model.flows import Planar, FirstLayer, Coupling, Scaling, Coupling_MLP

class PlanarFlow(nn.Module):  ##PlanarVI without VAEs
    """
    Stacking planar transformations
    """

    def __init__(self, K, z_size):
        super(PlanarFlow, self).__init__()

        # Initialize log-det-jacobian to zero
        self.log_det_j = 0.
        # Flow parameters
        flow = Planar
        q_0 = FirstLayer
        self.num_flows = K
        
        #First layer 
        flow_0 = q_0(z_size)
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
    def __init__(self, K, z_size):
        """Initialize a NICE flow.
        Args:
            K: number of coupling layers.
        """
        super(NICEFlow, self).__init__()
        # Flow parameters
        flow = Coupling
        q_0 = FirstLayer
        self.num_flows = K
        
        #First layer 
        flow_0 = q_0(z_size)
        self.add_module('flow_' + str(0), flow_0)
        
        # Normalizing flow layers
        for k in range(1,self.num_flows+1):
            flow_k = flow(z_size)
            scale_k = Scaling(z_size)
            self.add_module('flow_' + str(k), flow_k)
            self.add_module('scale_' + str(k), scale_k)

                    
        # self.scaling = Scaling(z_size)

    def forward(self, z):
        """Transformation f: X -> Z (inverse of g).
        Args:
            z: tensor in data space X.
        Returns:
            transformed tensor in latent space Z.
        """
        self.log_det_j = 0.
        for k in range(1, self.num_flows+1):           
            flow_k = getattr(self, 'flow_' + str(k))
            scale_k = getattr(self, 'scale_' + str(k))
            z, log_det_jacobian = scale_k(flow_k(z))
        # z, log_det_jacobian = self.scaling(z)
            self.log_det_j += log_det_jacobian                                                                

        return z, self.log_det_j

class NiceFlow(nn.Module):
    def __init__(self, K, z_size):
        """Initialize a NICE flow.
        Args:
            K: number of coupling layers.
        """
        super(NiceFlow, self).__init__()
        # Flow parameters
        flow = Coupling_MLP
        q_0 = FirstLayer
        self.num_flows = K
        
        #First layer 
        flow_0 = q_0(z_size)
        self.add_module('flow_' + str(0), flow_0)
        
        # Normalizing flow layers
        for k in range(1,self.num_flows+1):
            flow_k = flow(z_size, mid_dim=10, hidden=3)
            self.add_module('flow_' + str(k), flow_k)
                    
        self.scaling = Scaling(z_size)

    def forward(self, z):
        """Transformation f: X -> Z (inverse of g).
        Args:
            z: tensor in data space X.
        Returns:
            transformed tensor in latent space Z.
        """
        self.log_det_j = 0.
        flow_0 = getattr(self, 'flow_' + str(0))
        z, log_det_jacobian = flow_0(z)
        self.log_det_j += log_det_jacobian
        for k in range(1, self.num_flows+1):           
            flow_k = getattr(self, 'flow_' + str(k))
            z = flow_k(z)
        z, log_det_jacobian = self.scaling(z)
        self.log_det_j += log_det_jacobian                                                                

        return z, self.log_det_j

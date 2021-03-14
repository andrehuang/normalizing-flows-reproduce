#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import models.flowsVI as flows

class Planar(nn.Module):  ##PlanarVI without VAEs
    """
    Stacking planar transformations
    """

    def __init__(self, K: int = 6):
        super(Planar, self).__init__()

        # Initialize log-det-jacobian to zero
        self.log_det_j = 0.
        # Flow parameters
        flow = flows.Planar
        self.num_flows = K
           
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
            z, log_det_jacobian = flow_k(z)                                                                        
            self.log_det_j += log_det_jacobian


        return z, self.log_det_j 
    

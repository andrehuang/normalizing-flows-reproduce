import numpy as np
import torch
from utils.distributions import normal_dist

def binary_loss_function(target_distr, z_0, z_k, log_det_jacobians, beta=1.): ####should add z_mu, z_var when I also replace the planarFlow
        
        
    log_p_zk = -target_distr(z_k) # ln p(z_k): unnormalized target distribution 
    log_q_z0 = normal_dist(z_0, mean=torch.zeros(2), logvar=torch.zeros(2), dim=1) ##should add the possibility for different z_mu, z_var 

  
    kl = torch.sum(log_q_z0 - log_p_zk) - torch.sum(log_det_jacobians) #sum over batches     #beta?????
    
    batch_size = z_0.size(0) 
    kl = kl / batch_size
    
    return kl 

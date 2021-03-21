import numpy as np
import torch
import torch.nn as nn
from util.distributions import log_normal_dist

def binary_loss_function(x_recon, x, z_mu, z_var, z_0, z_k, log_det_jacobians, z_size, cuda, beta=1, summ = True, log_vamp_zk = None):
    """
    x_recon: shape: (batch_size, num_channels, pixel_width, pixel_height), i.e. for MNIST [batch_size, 1, 28, 28]
    x: shape (batchsize, num_channels, pixel_width, pixel_height), pixel values rescaled between [0, 1].
    z_mu: mean of z_0
    z_var: variance of z_0
    z_0: first stochastic latent variable
    z_k: last stochastic latent variable
    log_det_jacobians: log det jacobian
    beta: beta for annealing according to Equation 20
    log_vamp_zk: default None but log_p(zk) if VampPrior used
    the function returns: Free Energy Bound (ELBO), reconstruction loss, kl
    """
    batch_size = x.size(0) 
    if (summ == True):  ## Computes the binary loss function with summing over batch dimension 
        
        #Reconstruction loss: Binary cross entropy
        reconstruction_loss = nn.BCELoss(reduction='sum')
        log_p_xz = reconstruction_loss(x_recon, x)  #log_p(x|z_k)
        
        logvar=torch.zeros(batch_size, z_size) 
        
        if cuda == True:                      
            logvar=logvar.cuda()
            
        # calculate log_p(zk) under standard Gaussian unless log_p(zk) under VampPrior given
        if log_vamp_zk == None:
            log_p_zk = log_normal_dist(z_k, mean=0, logvar=logvar, dim=1) # ln p(z_k) = N(0,I)
        else:
            log_p_zk = log_vamp_zk
        
        log_q_z0 = log_normal_dist(z_0, mean=z_mu, logvar=z_var.log(), dim=1)
  
        kl = torch.sum(log_q_z0 - beta * log_p_zk) - torch.sum(log_det_jacobians) #sum over batches
        #Equation (20)
        elbo = beta * log_p_xz + kl
    
        elbo = elbo / batch_size
        log_p_xz = log_p_xz / batch_size
        kl = kl / batch_size
    
        return elbo, log_p_xz, kl 
    
    else:              ## Computes the binary loss function without summing over batch dimension (used during testing) 
        if len(log_det_jacobians.size()) > 1:
            log_det_jacobians = log_det_jacobians.view(log_det_jacobians.size(0), -1).sum(-1)

        reconstruction_loss = nn.BCELoss(reduction='none')
        log_p_xz = reconstruction_loss(x_recon.view(batch_size, -1), x.view(batch_size, -1))  #log_p(x|z_k)
        log_p_xz = torch.sum(log_p_xz, dim=1)
        
        logvar=torch.zeros(batch_size, z_size)  
        if cuda == True:                       
            logvar=logvar.cuda()
            
        # calculate log_p(zk) under standard Gaussian unless log_p(zk) under VampPrior given
        if log_vamp_zk == None:
            log_p_zk = log_normal_dist(z_k, mean=0, logvar=logvar, dim=1)
        else:
            log_p_zk = log_vamp_zk
        
        log_q_z0 = log_normal_dist(z_0, mean=z_mu, logvar=z_var.log(), dim=1)
        #Equation (20)
        elbo = log_q_z0 - beta * (log_p_zk - log_p_xz) - log_det_jacobians  

        return elbo, log_p_xz, (log_q_z0 - beta * log_p_zk - log_det_jacobians)

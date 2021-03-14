import numpy as np
import torch
import torch.nn as nn
from utils.distributions import normal_dist


def binary_loss_function(x_recon, x, z_mu, z_var, z_0, z_k, log_det_jacobians, beta=1. , summ = True):
    """
    :param recon_x: shape: (batch_size, num_channels, pixel_width, pixel_height), bernoulli parameters p(x=1)
    :param x: shape (batchsize, num_channels, pixel_width, pixel_height), pixel values rescaled between [0, 1].
    :param z_mu: mean of z_0
    :param z_var: variance of z_0
    :param z_0: first stochastic latent variable
    :param z_k: last stochastic latent variable
    :param ldj: log det jacobian
    :param beta: beta for kl loss
    :return: loss, ce, kl
    """
    batch_size = x.size(0) 
    if (summ == True):  ## Computes the binary loss function with summing over batch dimension
        
        #Reconstruction loss: Binary cross entropy
        reconstruction_loss = nn.BCELoss(reduction='sum')
        log_p_xz = reconstruction_loss(x_recon, x)  #log_p(x|z_k)
        
        
        log_p_zk = normal_dist(z_k, mean=0, logvar=z_var*0, dim=1) ###what shape does z_var has ???? # ln p(z_k) = N(0,I) 
        log_q_z0 = normal_dist(z_0, mean=z_mu, logvar=z_var.log(), dim=1)
  
        kl = torch.sum(log_q_z0 - log_p_zk) - torch.sum(log_det_jacobians) #sum over batches
        
        energy = log_p_xz + beta *kl
    
        energy = energy / batch_size
        log_p_xz = log_p_xz / batch_size
        kl = kl / batch_size
    
        return energy, log_p_xz, kl 
    
    else:             ## Computes the binary loss function without summing over batch dimension
        if len(log_det_jacobians.size()) > 1:
            log_det_jacobians = log_det_jacobians.view(log_det_jacobians.size(0), -1).sum(-1)

        reconstruction_loss = nn.BCELoss(reduction='none')
        log_p_xz = reconstruction_loss(x_recon.view(batch_size, -1), x.view(batch_size, -1))  #log_p(x|z_k)
        log_p_xz = torch.sum(log_p_xz, dim=1)
        
        log_p_zk = normal_dist(z_k, mean=0, logvar=z_var*0, dim=1)
        log_q_z0 = normal_dist(z_0, mean=z_mu, logvar=z_var.log(), dim=1)

        logs = log_q_z0 - log_p_zk

        loss = log_p_xz + beta * (logs - log_det_jacobians)

        return loss, log_p_xz, (logs - log_det_jacobians)


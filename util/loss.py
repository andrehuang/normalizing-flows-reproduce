import numpy as np
import torch
import torch.nn as nn
from util.distributions import log_normal_diag, log_normal_standard, log_bernoulli

## Reference: Sylvester flow; not tested yet
def binary_loss_function(recon_x, x, z_mu, z_var, z_0, z_k, ldj, beta=1.):
    """
    Computes the binary loss function while summing over batch dimension, not averaged!
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

    reconstruction_function = nn.BCELoss(reduction='sum')
    batch_size = x.size(0)
    bce = reconstruction_function(recon_x, x) # - N E_q0 [ ln p(x|z_k) ]

    #### Equation (20) in the paper ####
    log_p_zk = log_normal_standard(z_k, dim=1) # ln p(z_k)  (not averaged)
    log_q_z0 = log_normal_diag(z_0, mean=z_mu, log_var=z_var.log(), dim=1) # ln q(z_0)  (not averaged)
    summed_logs = torch.sum(log_q_z0 - beta * log_p_zk) # N E_q0[ ln q(z_0) - ln p(z_k) ]

    # sum over batches
    summed_ldj = torch.sum(ldj)

    
    # ldj = N E_q_z0[\sum_k log |det dz_k/dz_k-1| ]
    kl = (summed_logs - summed_ldj)
    loss = bce +  kl

    loss = loss / float(batch_size)
    rec = bce / float(batch_size)
    kl = kl / float(batch_size)

    return loss, rec, kl
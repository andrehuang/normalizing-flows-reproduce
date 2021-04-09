import torch
import torch.nn as nn
import models.flows as flows
from util.distributions import log_normal_dist
import math


class VAE(nn.Module):
    """
    The base VAE class.
    Can be used as a base class for VAE with normalizing flows.
    """

    def __init__(self, encoder, decoder, args):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

        # extract model settings from args
        self.z_size = args.z_size
        self.input_size = args.input_size # for reconstruction
        self.input_dim = args.input_dim # for defining p_mu function
        self.encoder_dim = args.encoder_dim # encoder feature_dim
        self.decoder_dim = args.decoder_dim # decoder feature_dim
        self.is_cuda = args.cuda

        self.mu = nn.Linear(self.encoder_dim, self.z_size)
        self.var = nn.Sequential(
            nn.Linear(self.encoder_dim, self.z_size),
            nn.Softplus(), # to make var positive
            nn.Hardtanh(min_val=1e-4, max_val=5.))

        if args.dataset == 'mnist':
            # Note MNIST is binarized; need add Sigmoid() function
            self.p_mu = nn.Sequential(
                    nn.Linear(self.decoder_dim, self.input_dim),
                    nn.Sigmoid() 
                )
        else:
            self.p_mu = nn.Linear(self.decoder_dim, self.input_dim)


        self.log_det_j = 0.
        self.num_pseudos = args.num_pseudos # for initialising pseudoinputs
            
    
    def init_pseudoinputs(self, pseudo_inputs):
        """
        Adds and initialises additional layer for pseudoinput generation
        pseudo_inputs: either random training data or None
        """
        
        self.pseudo_inputs = pseudo_inputs
        self.pseudo_inputs.requires_grad = False
        
        if pseudo_inputs is None:
            # initialise dummy inputs
            if self.is_cuda:
                self.dummy_inputs = torch.eye(self.num_pseudos).cuda()
            else:
                self.dummy_inputs = torch.eye(self.num_pseudos)
            self.dummy_inputs.requires_grad = False
            # initialise layers for learning pseudoinputs
            self.pseudo_layer = nn.Linear(self.num_pseudos, 784, bias=False)
            self.pseudo_layer.weight.data.normal_(-0.05, 0.01) #default in experiment parser
            self.pseudo_nonlin = nn.Hardtanh(min_val=0.0, max_val=1.0)
        elif self.is_cuda:
            self.pseudo_inputs = self.pseudo_inputs.cuda()
        
        
        
    def log_vamp_zk(self, zk):
        """
        Calculates log p(z_k) under VampPrior
        """
        
        # generate pseudoinputs from diagonal tensor
        if self.pseudo_inputs is None:
            pseudo_x = self.pseudo_nonlin(self.pseudo_layer(self.dummy_inputs))
        else:
            pseudo_x = self.pseudo_inputs
        
        # calculate VampPrior
        vamp_mu, vamp_logvar, _, _, _ = self.encode(pseudo_x)
        
        # expand
        zk_expanded = zk.unsqueeze(1)
        mus = vamp_mu.unsqueeze(0)
        logvars = vamp_logvar.unsqueeze(0)
        
        # calculate log p(z_k)
        log_per_pseudo = log_normal_dist(zk_expanded, mus, logvars, dim=2) - math.log(self.num_pseudos)
        log_total = torch.logsumexp(log_per_pseudo, 1)
        
        return log_total
    
    
    def reparameterize(self, mu, var):
        """
        Samples z from a multivariate Gaussian with diagonal covariance matrix using the
         reparameterization trick.
        """
        std = var.sqrt()
        eps = torch.randn_like(std)
        z = eps * std + mu
        return z

    def encode(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        mu = self.mu(h)
        var = self.var(h)
        return mu, var

    def decode(self, z):
        z = z.view(z.size(0), self.z_size, 1, 1)
        h = self.decoder(z)
        x_mean = self.p_mu(h)

        return x_mean.view(-1, *self.input_size)

    def forward(self, x):
        """
        Evaluates the model as a whole, encodes and decodes. Note that the log det jacobian is zero
         for a plain VAE (without flows), and z_0 = z_k = z.
        """

        # mean and variance of z
        z_mu, z_var = self.encode(x)
        # sample z
        z = self.reparameterize(z_mu, z_var)
        x_mean = self.decode(z)

        return x_mean, z_mu, z_var, self.log_det_j, z, z  # the lst three outputs are useless; only to match outputs of flowVAE


class PlanarVAE(VAE):
    """
    Variational auto-encoder with planar flows in the encoder.
    """

    def __init__(self, encoder, decoder, args):
        super(PlanarVAE, self).__init__(encoder, decoder, args)

        # Initialize log-det-jacobian to zero
        self.log_det_j = 0.

        # Flow parameters
        flow = flows.Planar
        self.num_flows = args.num_flows

        # Amortized flow parameters
        self.amor_u = nn.Linear(self.encoder_dim, self.num_flows * self.z_size)
        self.amor_w = nn.Linear(self.encoder_dim, self.num_flows * self.z_size)
        self.amor_b = nn.Linear(self.encoder_dim, self.num_flows)

        # Normalizing flow layers
        for k in range(self.num_flows):
            flow_k = flow()
            self.add_module('flow_' + str(k), flow_k)

        

    def encode(self, x):
        """
        Encoder that ouputs parameters for base distribution of z and flow parameters.
        """

        batch_size = x.size(0)

        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        mu = self.mu(h)
        var = self.var(h)

        u = self.amor_u(h).view(batch_size, self.num_flows, self.z_size, 1)
        w = self.amor_w(h).view(batch_size, self.num_flows, 1, self.z_size)
        b = self.amor_b(h).view(batch_size, self.num_flows, 1, 1)

        return mu, var, u, w, b

    def forward(self, x):
        """
        Forward pass with planar flows for the transformation z_0 -> z_1 -> ... -> z_k.
        Log determinant is computed as log_det_j = N E_q_z0[\sum_k log |det dz_k/dz_k-1| ].
        """

        if self.is_cuda:
            self.log_det_j = torch.zeros([x.shape[0]]).cuda()
        else:
            self.log_det_j = torch.zeros([x.shape[0]])


        z_mu, z_var, u, w, b = self.encode(x)

        # z_0
        z = [self.reparameterize(z_mu, z_var)]

        # Normalizing flows
        for k in range(self.num_flows):
            flow_k = getattr(self, 'flow_' + str(k))
            z_k, log_det_jacobian = flow_k(z[k], u[:, k, :, :], w[:, k, :, :], b[:, k, :, :])
            z.append(z_k)
            self.log_det_j += log_det_jacobian

        x_mean = self.decode(z[-1])

        return x_mean, z_mu, z_var, self.log_det_j, z[0], z[-1]

class NICEVAE(VAE):
    """
    Variational auto-encoder with NICE flows in the encoder.
    """

    def __init__(self, encoder, decoder, args):
        super(NICEVAE, self).__init__(encoder, decoder, args)

        # Initialize log-det-jacobian to zero
        self.log_det_j = 0.

        # Flow parameters
        self.num_flows = args.num_flows

        # NICE additive shift layers
        for k in range(self.num_flows):
            flow_k = flows.Coupling(in_out_dim=self.z_size, 
                     mid_dim=80, # to match the number of parameters in the NF flow
                     hidden=1)
            scale_k = flows.Scaling(self.z_size)
            self.add_module('flow_' + str(k), flow_k)
            self.add_module('scale_' + str(k), scale_k)

    def encode(self, x):
        """
        Encoder that ouputs parameters for base distribution of z and flow parameters.
        """

        batch_size = x.size(0)

        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        mu = self.mu(h)
        var = self.var(h)

        return mu, var

    def forward(self, x):
        """
        Forward pass with planar flows for the transformation z_0 -> z_1 -> ... -> z_k.
        Log determinant is computed as log_det_j = N E_q_z0[\sum_k log |det dz_k/dz_k-1| ].
        """

        if self.is_cuda:
            self.log_det_j = torch.zeros([x.shape[0]]).cuda()
        else:
            self.log_det_j = torch.zeros([x.shape[0]])


        z_mu, z_var= self.encode(x)

        # z_0 
        z_0 = self.reparameterize(z_mu, z_var)
        z = [z_0]

        # Normalizing flows
        for k in range(self.num_flows):
            flow_k = getattr(self, 'flow_' + str(k))
            scale_k = getattr(self, 'scale_' + str(k))
            z_k, log_det_jacobian = scale_k(flow_k(z[k]))
            z.append(z_k)
            self.log_det_j += log_det_jacobian

        x_mean = self.decode(z[-1])

        return x_mean, z_mu, z_var, self.log_det_j, z[0], z[-1]

class NICEVAE_amor(VAE):
    """
    Variational auto-encoder with NICE flows in the encoder.
    """

    def __init__(self, encoder, decoder, args):
        super(NICEVAE_amor, self).__init__(encoder, decoder, args)

        # Initialize log-det-jacobian to zero
        self.log_det_j = 0.

        # Flow parameters
        self.num_flows = args.num_flows

        # Amortized flow parameters
        self.amor_u = nn.Linear(self.encoder_dim, self.num_flows * (self.z_size//2))
        self.amor_w = nn.Linear(self.encoder_dim, self.num_flows * (self.z_size//2))
        self.amor_b = nn.Linear(self.encoder_dim, self.num_flows)
        self.amor_s = nn.Linear(self.encoder_dim, self.z_size)

        # NICE additive shift layers
        for k in range(self.num_flows):
            flow_k = flows.Coupling_amor()
            # scale_k = flows.Scaling(self.z_size)
            self.add_module('flow_' + str(k), flow_k)
            # self.add_module('scale_' + str(k), scale_k)
        
        self.scaling = flows.Scaling()
        
    def encode(self, x):
        """
        Encoder that ouputs parameters for base distribution of z and flow parameters.
        """

        batch_size = x.size(0)

        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        mu = self.mu(h)
        var = self.var(h)

        u = self.amor_u(h).view(batch_size, self.num_flows, self.z_size//2, 1)
        w = self.amor_w(h).view(batch_size, self.num_flows, 1, self.z_size//2)
        b = self.amor_b(h).view(batch_size, self.num_flows, 1, 1)
        s = self.amor_s(h).view(batch_size, self.z_size)

        return mu, var, u, w, b, s

        

    def forward(self, x):
        """
        Forward pass with planar flows for the transformation z_0 -> z_1 -> ... -> z_k.
        Log determinant is computed as log_det_j = N E_q_z0[\sum_k log |det dz_k/dz_k-1| ].
        """

        if self.is_cuda:
            self.log_det_j = torch.zeros([x.shape[0]]).cuda()
        else:
            self.log_det_j = torch.zeros([x.shape[0]])


        z_mu, z_var, u, w, b, s = self.encode(x)

        # z_0 
        z_0 = self.reparameterize(z_mu, z_var)
        z = [z_0]

        # Normalizing flows
        for k in range(self.num_flows):
            flow_k = getattr(self, 'flow_' + str(k))
            # scale_k = getattr(self, 'scale_' + str(k))
            z_k = flow_k(z[k], u[:, k, :, :], w[:, k, :, :], b[:, k, :, :])
            z.append(z_k)
            self.log_det_j += 0
        z_k, log_det_jacobian = self.scaling(z_k, s)
        z.append(z_k)
        self.log_det_j += log_det_jacobian

        x_mean = self.decode(z[-1])

        return x_mean, z_mu, z_var, self.log_det_j, z[0], z[-1]
   
class Sylvester_ortho_VAE(VAE):
    """
    Variational auto-encoder with orthogonal flows in the encoder.
    """

    def __init__(self, encoder, decoder, args):
        super(Sylvester_ortho_VAE, self).__init__(encoder, decoder, args)

        # Flow parameters
        flow = flows.Sylvester
        self.num_flows = args.num_flows
        self.M = args.num_orthonormal_vec
    
        if (self.M > self.z_size) or (self.M <= 0): #z_size: number of stochastic hidden units
            raise ValueError('ERROR: The number of orthonomal vectors (M) must be positive and smaller than the dimension of z (D)')
        
        # Orthogonalization parameters
        self.epsilon = 1.e-6
        self.steps = 100
        self.invertibility = args.syl_ortho_invertible
        
        self.identity = torch.eye(self.M, self.M)
        self.identity = self.identity.unsqueeze(0)
        
        # torch used to make R and R_tilde upper triangluar 
        self.upper_triang = torch.triu(torch.ones(self.M, self.M))
        self.upper_triang = self.upper_triang.unsqueeze(0).unsqueeze(3)
        
        if self.is_cuda:
            self.identity = self.identity.cuda()
            self.upper_triang =  self.upper_triang.cuda()
                
        # Amortized flow parameters
        self.amor_Q = nn.Linear(self.encoder_dim, self.num_flows * self.z_size * self.M)
        self.amor_b = nn.Linear(self.encoder_dim, self.num_flows * self.M)
        self.amor_R = nn.Linear(self.encoder_dim, self.num_flows * self.M * self.M)
        self.amor_R_tilde = nn.Linear(self.encoder_dim, self.num_flows * self.M * self.M)
               
        #for the flow to be invertible we need R*R_tilde>-1 - default: self.invertibility = False
        if self.invertibility:
            self.amor_diag1 = nn.Sequential(nn.Linear(self.encoder_dim, self.num_flows * self.M), nn.Tanh())
            self.amor_diag2 = nn.Sequential(nn.Linear(self.encoder_dim, self.num_flows * self.M), nn.Tanh())
        
        # Normalizing flow layers
        for k in range(self.num_flows):
            flow_k = flow(self.M)
            self.add_module('flow_' + str(k), flow_k)

    def get_orthonormal(self, Q):
        """
        Iterative orthogonalization of Q(0) up to Q(self.steps) (in parallel for all flows)
        shape Q = (batch_size * num_flows, z_size, M)
        """
        
        # Reshape from (batch_size, num_flows * z_size * M) to (batch_size * num_flows, z_size, M)
        Q = Q.view(-1, self.z_size , self.M)
        norm = torch.norm(Q, p=2, dim=[1,2], keepdim=True)
        Q_k = torch.div(Q, norm)
            
        # Iterative orthogonalization
        for k in range(self.steps):
            prod = torch.bmm(Q_k.transpose(2, 1), Q_k)
            prod = self.identity + 0.5 * (self.identity - prod)
            Q_k = torch.bmm(Q_k, prod)
            
            # Checking convergence - ||Q^T Q - I||<= self.epsilon
            prod = torch.bmm(Q_k.transpose(2, 1), Q_k) - self.identity
            norm = torch.sum(torch.norm(prod, p='fro', dim=2) ** 2, dim=1)
            norm = torch.sqrt(norm)
            max_norm = torch.max(norm)
            if max_norm <= self.epsilon:
                break
        
        if max_norm > self.epsilon:
            print('WARNING: orthogonalization has not converged with repect to the threshold (epsilon)')

        Q_k = Q_k.view(-1, self.num_flows, self.z_size, self.M)

        return Q_k.transpose(1, 0)

    def encode(self, x):
        """
        Outputs the parameters of the base distribution q_0 and the flow parameters Q,R,R_tilde, b
        """

        batch_size = x.size(0)

        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        mu = self.mu(h)
        var = self.var(h)
        
        #Return amortized Q, b, R, R_tilde
        Q = self.amor_Q(h)
        b = self.amor_b(h)
        R = self.amor_R(h)
        R_tilde = self.amor_R_tilde(h)
        
        R = R.view(batch_size, self.M, self.M, self.num_flows)
        R_tilde = R_tilde.view(batch_size, self.M, self.M, self.num_flows)

        R = R * self.upper_triang
        R_tilde = R_tilde * self.upper_triang       
        
        if self.invertibility:
            diag1 = self.amor_diag1(h)
            diag2 = self.amor_diag2(h)
            diag1 = diag1.view(batch_size, self.M, self.num_flows)
            diag2 = diag2.view(batch_size, self.M, self.num_flows)
            R[:, range(self.M), range(self.M), :] = diag1
            R_tilde[:, range(self.M), range(self.M), :] = diag2
            
        # Resize b from shape [batch_size, M * num_flows] to [batch_size, M, num_flows]
        b = b.view(batch_size, self.M, self.num_flows)
      
        return mu, var, Q, R, R_tilde, b

    def forward(self, x):

        if self.is_cuda:
            log_det_j = torch.zeros([x.shape[0]]).cuda()
        else:
            log_det_j = torch.zeros([x.shape[0]])

        z_mu, z_var, Q, R, R_tilde, b = self.encode(x)

        # Orthogonalize all q matrices
        Q_orthonormal = self.get_orthonormal(Q)

        # z_0
        z = [self.reparameterize(z_mu, z_var)]

        # Normalizing flows
        for k in range(self.num_flows):

            flow_k = getattr(self, 'flow_' + str(k))
            z_k, log_det_jacobian = flow_k(z[k], Q_orthonormal[k, :, :, :], R[:, :, :, k], R_tilde[:, :, :, k], b[:, :, k])

            z.append(z_k)
            log_det_j += log_det_jacobian

        x_mean = self.decode(z[-1])

        return x_mean, z_mu, z_var, log_det_j, z[0], z[-1]
    
    
class RealNVPVAE(VAE):
    def __init__(self, encoder, decoder, args):
        super(RealNVPVAE, self).__init__(encoder, decoder, args)
        self.log_det_j = 0.
        self.num_flows = args.num_flows
        for k in range(self.num_flows):
            flow_k = flows.AffineCoupling(self.z_size, 80, 1)
            self.add_module('flow_' + str(k), flow_k)
            
    def encode(self, x):
        batch_size = x.size(0)
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        mu = self.mu(h)
        var = self.var(h)
        return mu, var
    
    def forward(self, x):
        self.log_det_j = torch.zeros([x.shape[0]])
        z_mu, z_var= self.encode(x)
        z = [self.reparameterize(z_mu, z_var)]
        for k in range(self.num_flows):
            flow_k = getattr(self, 'flow_' + str(k))
            z_k, log_det_jacobian = flow_k(z[k])
            z.append(z_k)
            # print("z", str(k))
            # print(z_k)
            # print(z_k.shape)
            self.log_det_j += log_det_jacobian
        x_mean = self.decode(z[-1])
        return x_mean, z_mu, z_var, self.log_det_j, z[0], z[-1]

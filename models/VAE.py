import torch
import torch.nn as nn
import models.flows as flows


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
        return mean, var

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

        self.log_det_j = 0.

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
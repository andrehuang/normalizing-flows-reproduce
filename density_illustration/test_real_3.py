from __future__ import print_function
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import random
import torch
import torch.utils.data
from torch.autograd import Variable
import numpy as np

import math

class FirstLayer(nn.Module):
    def __init__(self, z_size):
        super().__init__()
        self.mu = nn.Parameter(torch.zeros(z_size)).requires_grad_(True)
        self.logvar = nn.Parameter(torch.zeros(z_size)).requires_grad_(True)

    def forward(self, z):
        f_z = self.mu + self.logvar.exp() * z
        self.sum_log_abs_det_jacobians = self.logvar.sum() #det of a diagonal matrix

        return f_z, self.sum_log_abs_det_jacobians * torch.ones(1,z.shape[0])


class AffineCoupling(nn.Module):
    def __init__(self, input_output_dim, mid_dim, hidden_dim, mask):
        super(AffineCoupling, self).__init__()
        self.input_output_dim = input_output_dim
        self.mid_dim = mid_dim
        self.hidden_dim = hidden_dim
        self.mask = mask
        self.s = nn.Sequential(nn.Linear(input_output_dim//2, mid_dim), nn.Tanh(), nn.Linear(mid_dim, mid_dim), nn.Tanh(), nn.Linear(mid_dim, input_output_dim//2))
        self.t = nn.Sequential(nn.Linear(input_output_dim//2, mid_dim), nn.Tanh(), nn.Linear(mid_dim, mid_dim), nn.Tanh(), nn.Linear(mid_dim, input_output_dim//2))

    def forward(self, x):
        d = self.input_output_dim//2
        # x1, x2 = x[:, :d], x[:, d:] # earlier split

        x1, x2 = x[:, ::2], x[:, 1::2]
        # skipping 2 elements x1 becomes values at 0,2,4 and x2 becomes 1,5,7..checkerboard masking
        if self.mask:
            x1, x2 = x2, x1

        scale = self.s(x1)
        translate = self.t(x1)

        z1 = x1
        z2 = x2 * torch.exp(scale)
        z3 = translate
        z4 = z2 + z3

        if self.mask:
            z1, z4 = z4, z1

        z = torch.cat((z1, z4), dim=1)
        log_det_j = scale.sum(-1)
        return z, log_det_j


class RealFlow(torch.nn.Module):
    def __init__(self, K, z_size):
        super(RealFlow, self).__init__()
        flow = AffineCoupling
        q_0 = FirstLayer
        self.num_flows = K

        #First layer
        flow_0 = q_0(z_size)
        self.add_module('flow_' + str(0), flow_0)


        # Normalizing flow layers
        for k in range(1,self.num_flows+1):
            mask = 0 if k%2==0 else 1
            flow_k = flow(z_size, mid_dim=10, hidden_dim=3, mask=0)
            self.add_module('flow_' + str(k), flow_k)

    def forward(self,z):
        self.log_det_j = 0.
        flow_0 = getattr(self, 'flow_' + str(0))
        z, log_det_jacobian = flow_0(z)
        self.log_det_j += log_det_jacobian
        for k in range(1, self.num_flows+1):
            mask = 0 if k%2==0 else 1
            flow_k = getattr(self, 'flow_' + str(k))
            z, log_det_jacobian = flow_k(z)
        self.log_det_j += log_det_jacobian
        return z, self.log_det_j


def target_distribution(name):

    w1 = lambda z: torch.sin(2 * np.pi * z[:, 0] / 4)
    w2 = lambda z: 3 * torch.exp(-0.5 * ((z[:, 0] - 1) / 0.6) ** 2)
    w3 = lambda z: 3 * torch.sigmoid((z[:, 0] - 1) / 0.3)

    if name == "1":

        u = lambda z: 0.5 * ((torch.norm(z, p=2, dim=1) - 2) / 0.4)**2 - \
                      torch.log(torch.exp(-0.5*((z[:,0] - 2) / 0.6)**2) + torch.exp(-0.5*((z[:,0] + 2) / 0.6)**2) + 1e-10)

    elif name == "2":

        u = lambda z: 0.5 * ((z[:,1] - w1(z)) / 0.4)**2

    elif name == "3":

        u = lambda z: - torch.log(torch.exp(-0.5*((z[:,1] - w1(z))/0.35)**2) + torch.exp(-0.5*((z[:,1] - w1(z) + w2(z))/0.35)**2) + 1e-10)

    elif name == "4":

        u = lambda z: - torch.log(torch.exp(-0.5*((z[:,1] - w1(z))/0.4)**2) + torch.exp(-0.5*((z[:,1] - w1(z) + w3(z))/0.35)**2) + 1e-10)

    return u

def plot_true_density(target_density, axlim, ax=None):
    n = 1000
    x = torch.linspace(-axlim, axlim, n)
    X, Y = torch.meshgrid(x, x)
    Z = torch.stack((X.flatten(), Y.flatten()), dim=-1)
    shape = X.shape

    ax.pcolormesh(X, Y, torch.exp(-target_density(Z)).view(shape,shape), cmap=plt.cm.jet)
    ax.set_aspect(1.) #sets y-axis, x-axis ratio
    plt.setp(ax, xticks=[], yticks=[])


def plot_flow_density(model, axlim, ax=None):
    n = 1000
    x = torch.linspace(-axlim, axlim, n)
    X, Y = torch.meshgrid(x, x)
    Z = torch.stack((X.flatten(), Y.flatten()), dim=-1)

    #plot posterior approximation
    z_k, sum_log_det_j = model(Z)

    q_0 = torch.distributions.MultivariateNormal(torch.zeros(2), torch.eye(2)) ###z_size and mu, var different that 0, I
    # Equation (7)
    log_q_k = q_0.log_prob(Z) - sum_log_det_j
    q_k = torch.exp(log_q_k)

    ax.set_xlim(-axlim, axlim)
    ax.set_ylim(-axlim, axlim)

    ax.pcolormesh(
        z_k[:, 0].detach().numpy().reshape(n,n), # detach().numpy() is required because zk is a tensor that requires grad
        z_k[:, 1].detach().numpy().reshape(n,n),
        q_k.detach().numpy().reshape(n,n),
        cmap=plt.cm.jet
    )
    ax.set_aspect(1)
    plt.setp(ax, xticks=[], yticks=[])
    ax.set_facecolor(plt.cm.jet(0.))


def plot_comparison(model, target_distr, flow_length):
    axlim = 4
    fig, axes = plt.subplots(ncols=2, nrows=1, figsize=[20, 5])

    # plot target density
    target = target_distribution(target_distr)
    plot_true_density(target, axlim=axlim, ax=axes[0])
    axes[0].set_title(f"True density of potential number '{target_distr}'", size=14)

    #plot approximated density
    plot_flow_density(model, axlim=axlim, ax=axes[1])
    axes[1].set_title(f"Estimated density of potential number '{target_distr}'", size=14)


def plot_all_targets():
    target_distributions = ["1", "2", "3", "4"]
    fig, axes = plt.subplots(ncols=len(target_distributions), nrows=1, figsize=[25, 15])
    for i, distr in enumerate(target_distributions):
        axlim = 4
        density = target_distribution(distr)
        plot_true_density(density, axlim=axlim, ax=axes[i])
        axes[i].set_title(f"Name: '{distr}'", size=14)


# log N(x| mean, var) = -log sqrt(2pi) -0.5 log var - 0.5 (x-mean)(x-mean)/var

def normal_dist(x, mean, logvar, dim):
    log_norm = -0.5 * (logvar + (x - mean) * (x - mean) * logvar.exp().reciprocal())

    return torch.sum(log_norm, dim) ##why??######



#Loss function, Equation (15)

def binary_loss_function(model, target_distr, z_0, z_k, log_det_jacobians, beta=1.):

    log_p_zk = - target_distr(z_k) # ln p(z_k): unnormalized target distribution
    log_q_z0 = normal_dist(z_0, mean=torch.zeros(2), logvar=torch.zeros(2), dim=1)

    # kl =  - torch.sum(log_det_jacobians)  - beta * torch.sum(log_p_zk)#sum over batches
    kl = (torch.sum(log_q_z0 - 2*log_p_zk)  - torch.sum(log_det_jacobians) )#sum over batches
    kl = kl / z_0.size(0)

    return kl

#Training function

def train(model, opt, num_batches, batch_size, density):

    model.train()
    for  batch_num in range(1, num_batches + 1):

        # Get batch from N(0,I).
        batch = torch.zeros(size=(batch_size, 2)).normal_(mean=0, std=1)
        # Pass batch through flow.
        zk, log_jacobians = model(batch)

        # Compute loss under target distribution.
        # beta = min(1, batch_num/10000)
        loss = binary_loss_function(model, density, batch, zk, log_jacobians)

        #Optimization step, Backpropagation
        opt.zero_grad()
        loss.backward() #  Why second backward will fail? (Add constant, not calculated quantity to log_det_J)
        opt.step()
        loss = loss.item()

        if batch_num%5000 == 0:
            print('Batch_num: {:3d}/ {:3d} loss: {:.4f}'.format(batch_num, num_batches, loss))

    return loss

def run(target, flow_type, num_flows, num_batches, batch_size, lr, resume=False):

    print("Density appoximation for test energy function " + target)
    loss = []
    for K in num_flows:
        flow_length = K
        print("Number of flows:")
        print(flow_length)

        # Initializations
        if flow_type == "planar":
            model = PlanarFlow(K=flow_length, z_size=z_size)
        elif flow_type == "NICE":
            # model = NICEFlow(K=flow_length, z_size=z_size)
            model = NiceFlow(K=flow_length, z_size=z_size)
        elif flow_type == 'real':
            model = RealFlow(K=flow_length,z_size=z_size)

        model_name = './ckpts/' + "target_" + target + "_" + flow_type + str(flow_length) + '.pth'
        fig_name = target + flow_type + str(flow_length) + '.png'
        if resume:
            try:
                model.load_state_dict(torch.load(model_name))
            except:
                print("Not finding saved model to resume")

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        density = target_distribution(target)

        #Training for the target distribution
        train_loss = train(model, optimizer, num_batches, batch_size, density)
        torch.save(model.state_dict(), model_name)
        plot_comparison(model, target, flow_length, fig_name)

        loss.append(train_loss)

    return model, loss

# Define parameters
z_size = 2
num_batches = 1000 * 200
batch_size = 500
num_flows = [32]
learning_rate = 1e-4

for target_distr in ["4", "1"]:
    model, loss = run(target_distr, 'real', num_flows, num_batches, batch_size, lr = learning_rate, resume=False)





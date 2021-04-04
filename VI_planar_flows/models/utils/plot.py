import torch
import numpy as np
import matplotlib.pyplot as plt
from target_distribution import target_distribution

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
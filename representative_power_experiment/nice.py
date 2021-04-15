import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from utils.target_distribution import target_distribution
from utils.distributions import normal_dist
from utils.plot import plot_all_targets, plot_comparison
from models.flows import Planar, FirstLayer, Coupling, Scaling
from models.model import PlanarFlow, NICEFlow, NiceFlow

import numpy as np
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
    model, loss = run(target_distr, 'NICE', num_flows, num_batches, batch_size, lr = learning_rate, resume=False)
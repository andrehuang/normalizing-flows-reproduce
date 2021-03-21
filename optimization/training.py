import torch
import numpy as np
from optimization.loss import binary_loss_function
from util.log_likelihood import calculate_likelihood

def train(epoch, train_loader, model, opt, args):

    model.train()
    train_loss = np.zeros(len(train_loader))
    num_data = 0
    index = -1
    
    for data, _ in train_loader:     
        
        index += 1
        # Beta-annealing
        beta = min(0.01 + ((epoch - 1) * 500 + index + 1) / 10000 , 1)
        
        if args.cuda:
            data = data.cuda()

        data = data.view(-1, *args.input_size) 

        #Pass throught the VAE + flows 
        x_mean, z_mu, z_var, log_det_jacobians, z0, zk = model(data)
        
        #Compute VampPrior if used
        log_vamp_zk = model.log_vamp_zk(zk) if args.vampprior else None
        
        #Compute the loss 
        loss, rec, kl = binary_loss_function(x_mean, data, z_mu, z_var, z0, zk, log_det_jacobians,
                                             args.z_size, beta = beta, log_vamp_zk = log_vamp_zk)

        #Optimization step, Backpropagation
        opt.zero_grad()
        loss.backward()
        train_loss[index] = loss.item()
        opt.step()

        num_data += len(data)

        if index % args.log_interval == 0:
            print('Epoch: {:3d} [{:5d}/{:5d} ({:2.0f}%)]  \tLoss: {:11.6f}\trec: {:11.6f}\tkl: {:11.6f}'.format(
                    epoch, num_data, len(train_loader.sampler), 100. * index / len(train_loader),
                    loss.item(), rec.item(), kl.item()))

    print('====> Epoch: {:3d} Average train loss: {:.4f}'.format(
            epoch, train_loss.sum() / len(train_loader)))
    print('beta = {:5.4f}'.format(beta))
        
    return train_loss


def evaluate(data_loader, model, args, testing=False):
    
    model.eval()
    loss = 0.

    for data, _ in data_loader:  

        if args.cuda:
            data = data.cuda()

        data = data.view(-1, *args.input_size)
        
        x_mean, z_mu, z_var, log_det_jacobians, z0, zk = model(data)
        
        #Compute VampPrior if used
        log_vamp_zk = model.log_vamp_zk(zk) if args.vampprior else None
        
        batch_loss, rec, kl = binary_loss_function(x_mean, data, z_mu, z_var, z0, zk, log_det_jacobians, 
                                                   args.z_size, log_vamp_zk = log_vamp_zk)
        loss += batch_loss.item()

    loss /= len(data_loader)

    # if we are in the testing period - Compute log-likelihood: -logp(x)
    if testing:
        test_data = data_loader.dataset.tensors[0]

        if args.cuda:
            test_data = test_data.cuda()

        print('Computing log-likelihood on test set')

        model.eval()

        log_likelihood = calculate_likelihood(test_data, model, args, S=5000, MB=1000) #calculate the true marginal likelihood by IS 
                                                                                       #using S samples from the inference network
        print('====> Test set loss: {:.4f}'.format(loss))
        print('====> Test set log-likelihood: {:.4f}'.format(log_likelihood))
        return log_likelihood
    else:
        
        print('====> Validation set loss: {:.4f}'.format(loss))
        return loss

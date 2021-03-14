# normalizing-flows-reproduce
ATML Group 10

## Update March. 7
by Haiwen

Basic implementations
- [x] Planar flow
- [x] VAE, and VAE with planar flow
- [x] Binarized MNIST dataset and data loader
- [] Loss function 
- [] Training and testing of MNIST

To Angeliki:
1. I have tested the flow and VAE with flow implementation. You can continue from that. I wrote a few lines of examples on how to define these models.

2. The loss.py file is copied from Sylvester flow. But I haven't tried it or tested it. I think you can write your own version of loss function using that as a reference.

3. You also need to write training and testing code of MNIST. I think you can work on *Colab* and forget about that `main_VAE.py` file.

I expect that it would not be easy to match the results in the paper. We will need to carefully check whether we match the implementations in the paper, and tuning the unmentioned model architectures and hyperparameters. This would take quite some time. We will need at least one more person to work on reproducing the original paper.



## Update March. 14
by Angeliki

**Basic implementations*
- [x] Loss function 
- [x] Training and testing of MNIST
- [x] Script for running everything -> main_experiment_VAE

I have tested the optimization/loss.py, optimization/training.py, util/distributions.py and util/log_likelihood.py and main_experiment_VAE.py implementations.

util/log_likelihood.py which calculates the true marginal likelihood by importance sampling was adopted as it was from the reference code (i.e. Sylvester flows) - if someone wants, they can write their own version of it.

**Density estimation/ Reproduce Fig.3*

I have implemented the code for the reproduction of Fig. 3 (a), (b) of the paper. I have uploaded the files accompanied by a notebook in the folder VI_planar_flows. I will fill this report here explaining more tomorrow, since I think this part is not related to what Ella will start doing tomorrow (and it's quite late in the night xD). 
reference: https://github.com/e-hulten/planar-flows

In relation to the MNIST experiment: 

- I have tested the MNIST training/testing for length of flows K=4 and K=10 for 1000 epochs (500k parameter updates). The results are shown in the test_colab notebook. For the case of K=10, a cuda-related error (I think it's Colab related) terminated the program (in epoch 754) and because of that I didn't check the final ELBO and the final lnp(x) values. However, for many epochs before and until the 754th, the algorithm was stack at a validation set loss around 104, whereas we want it (the test set loss, but they are close) to be around 93-94, according to Fig 4a. For the case of K=4, ELBO was around 95 and lnp(x) around 89, which are the needed values. 

For the K=10 case, I expect that with the change of the architecture of the encoder/decoder the values of ELBO and lnp(x) will get better. 

Also, I noticed that when using our code and functions and only use the reference (i.e. Sylvester) code for the models.flows we get better results for the K=10 case (Validation set loss: 91.8). The only difference in their implementantion of models.flows is the use of u_hat instead of u.


- For number of flows K=10, I encounter an error which, after investigating its cause, seems to me that is VAE architecture related. I describe the procedure for the reproduction of the error and the reason why I believe it's VAE architecture related in detail in the file Errors_README.

To Ella: you can start adjusting the architecture/hyperparameters of encoder/decoder of the VAE. 



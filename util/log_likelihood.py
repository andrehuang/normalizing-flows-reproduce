from __future__ import print_function
import numpy as np
from scipy.special import logsumexp
from optimization.loss import binary_loss_function

## Code adopted from
# https://github.com/riannevdberg/sylvester-flows/blob/master/utils/log_likelihood.py

def calculate_likelihood(X, model, args, S=5000, MB=500):

    # set auxiliary variables for number of training and test sets
    N_test = X.size(0)

    X = X.view(-1, *args.input_size)

    likelihood_test = []

    if S <= MB:
        R = 1
    else:
        R = S // MB
        S = MB

    for j in range(N_test):
        if j % 100 == 0:
            print('Progress: {:.2f}%'.format(j / (1. * N_test) * 100))

        x_single = X[j].unsqueeze(0)

        a = []
        for r in range(0, R):
            # Repeat it for all training points
            x = x_single.expand(S, *x_single.size()[1:]).contiguous()

            x_mean, z_mu, z_var, ldj, z0, zk = model(x)

            a_tmp, _ , _ = binary_loss_function(x_mean, x, z_mu, z_var, z0, zk, ldj, summ=False)

            a.append(-a_tmp.cpu().data.numpy())

        # calculate max
        a = np.asarray(a)
        a = np.reshape(a, (a.shape[0] * a.shape[1], 1))
        likelihood_x = logsumexp(a)
        likelihood_test.append(likelihood_x - np.log(len(a)))

    likelihood_test = np.array(likelihood_test)

    nll = -np.mean(likelihood_test)

    return nll

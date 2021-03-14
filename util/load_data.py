import torch
import torch.utils.data as data_utils
import pickle
from scipy.io import loadmat

import numpy as np

import os

## Code adopted from
# https://github.com/riannevdberg/sylvester-flows/blob/master/utils/load_data.py


def load_mnist(args):
    """
    Dataloading function for static mnist. Outputs image data in vectorized form: each image is a vector of size 784
    """
    args.input_size = [1, 28, 28]
    args.input_dim = 784

    # start processing
    def lines_to_np_array(lines):
        return np.array([[int(i) for i in line.split()] for line in lines])

    with open(os.path.join('data', 'MNIST_binary', 'binarized_mnist_train.amat')) as f:
        lines = f.readlines()
    x_train = lines_to_np_array(lines).astype('float32')
    with open(os.path.join('data', 'MNIST_binary', 'binarized_mnist_valid.amat')) as f:
        lines = f.readlines()
    x_val = lines_to_np_array(lines).astype('float32')
    with open(os.path.join('data', 'MNIST_binary', 'binarized_mnist_test.amat')) as f:
        lines = f.readlines()
    x_test = lines_to_np_array(lines).astype('float32')

    # shuffle train data
    np.random.shuffle(x_train)

    # idle y's
    y_train = np.zeros((x_train.shape[0], 1))
    y_val = np.zeros((x_val.shape[0], 1))
    y_test = np.zeros((x_test.shape[0], 1))

    # pytorch data loader
    train = data_utils.TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    train_loader = data_utils.DataLoader(train, batch_size=args.batch_size, shuffle=True)

    validation = data_utils.TensorDataset(torch.from_numpy(x_val).float(), torch.from_numpy(y_val))
    val_loader = data_utils.DataLoader(validation, batch_size=args.batch_size, shuffle=False)

    test = data_utils.TensorDataset(torch.from_numpy(x_test).float(), torch.from_numpy(y_test))
    test_loader = data_utils.DataLoader(test, batch_size=args.batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, args

def load_dataset(args):

    if args.dataset == 'mnist':
        train_loader, val_loader, test_loader, args = load_mnist(args)

    return train_loader, val_loader, test_loader, args

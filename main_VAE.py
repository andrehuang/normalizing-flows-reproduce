# MNIST train and test
import argparse
import time
import os
import torch
import torch.utils.data
import torch.optim as optim
import numpy as np
import math
import random


import models.VAE as VAE, PlanarVAE
from models.model import MLP_encoder, MLP_decoder
from util.load_data import load_dataset
from util.loss import binary_loss_function # Not tested yet. To Angeliki: you can write your own version.


parser = argparse.ArgumentParser(description='Variational Inference with Normalizing flows')

parser.add_argument('-d', '--dataset', type=str, default='mnist', choices=['mnist'],
                    metavar='DATASET',
                    help='Dataset choice. Currently only support MNIST.')

parser.add_argument('-nc', '--no_cuda', action='store_true', default=False,
                    help='disables CUDA training')

parser.add_argument('--manual_seed', type=int, default=42, help='manual seed, if not given resorts to random seed.')

parser.add_argument('-li', '--log_interval', type=int, default=10, metavar='LOG_INTERVAL',
                    help='how many batches to wait before logging training status')

parser.add_argument('-od', '--out_dir', type=str, default='snapshots', metavar='OUT_DIR',
                    help='output directory for model snapshots etc.')

# optimization settings
parser.add_argument('-e', '--epochs', type=int, default=2000, metavar='EPOCHS',
                    help='number of epochs to train (default: 2000)')
parser.add_argument('-bs', '--batch_size', type=int, default=100, metavar='BATCH_SIZE',
                    help='input batch size for training (default: 100)')
parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4, metavar='LEARNING_RATE',
                    help='learning rate')
parser.add_argument('-f', '--flow', type=str, default='planar', choices=['planar'],
                    help="""Type of flows to use. Only planar flow is supported now.""")
parser.add_argument('-nf', '--num_flows', type=int, default=4,
                    metavar='NUM_FLOWS', help='Number of flow layers, ignored in absence of flows')
parser.add_argument('--z_size', type=int, default=64, metavar='ZSIZE',
                    help='how many stochastic hidden units')
parser.add_argument('--encoder_dim', type=int, default=256, metavar='ESIZE',
                    help='output feature dim of encoder')
parser.add_argument('--decoder_dim', type=int, default=256, metavar='DSIZE',
                    help='output feature dim of decoder')                    
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


def main(args):
    print('\nMODEL SETTINGS: \n', args, '\n')
    print("Random Seed: ", args.manual_seed)

    # set snapshot dir

    train_loader, val_loader, test_loader, args = load_dataset(args)

    encoder = MLP_encoder(args)
    decoder = MLP_decoder(args)
    model = VAE.PlanarVAE(encoder, decoder, args)
    if args.cuda:
        print("Model on GPU")
        model.cuda()
    print(model)

    optimizer = optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    ## Train

    ## Test

    ## Save model to snapshot dir

if __name__ == "__main__":
    main(args)



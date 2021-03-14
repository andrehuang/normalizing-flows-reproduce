import argparse
import time
import torch
import torch.utils.data
import torch.optim as optim
import numpy as np
import math
import random

import models.VAE as VAE
from models.model import MLP_encoder, MLP_decoder
from optimization.training import train, evaluate
from util.load_data import load_dataset
from util.plotting import plot_training_curve


parser = argparse.ArgumentParser(description='PyTorch Sylvester Normalizing flows')

parser.add_argument('-d', '--dataset', type=str, default='mnist', choices=['mnist'],
                    metavar='DATASET',
                    help='Dataset choice.')

parser.add_argument('-nc', '--no_cuda', action='store_true', default=False,
                    help='disables CUDA training')

parser.add_argument('--manual_seed', type=int, help='manual seed, if not given resorts to random seed.')

parser.add_argument('-li', '--log_interval', type=int, default=10, metavar='LOG_INTERVAL',
                    help='how many batches to wait before logging training status')

parser.add_argument('-od', '--out_dir', type=str, default='snapshots', metavar='OUT_DIR',
                    help='output directory for model snapshots etc.')

fp = parser.add_mutually_exclusive_group(required=False)
fp.add_argument('-te', '--testing', action='store_true', dest='testing',
                help='evaluate on test set after training')
fp.add_argument('-va', '--validation', action='store_false', dest='testing',
                help='only evaluate on validation set')
parser.set_defaults(testing=True)

# optimization settings
parser.add_argument('-e', '--epochs', type=int, default=1000, metavar='EPOCHS',
                    help='number of epochs to train (default: 2000)')
parser.add_argument('-es', '--early_stopping_epochs', type=int, default=100, metavar='EARLY_STOPPING',
                    help='number of early stopping epochs')

parser.add_argument('-bs', '--batch_size', type=int, default=100, metavar='BATCH_SIZE',
                    help='input batch size for training (default: 100)')
parser.add_argument('-lr', '--learning_rate', type=float, default=0.0005, metavar='LEARNING_RATE',
                    help='learning rate')

parser.add_argument('-w', '--warmup', type=int, default=100, metavar='N',
                    help='number of epochs for warm-up. Set to 0 to turn warmup off.')
parser.add_argument('--max_beta', type=float, default=1., metavar='MB',
                    help='max beta for warm-up')
parser.add_argument('--min_beta', type=float, default=0.0, metavar='MB',
                    help='min beta for warm-up')
parser.add_argument('-f', '--flow', type=str, default='planar', choices=['planar'])
parser.add_argument('-nf', '--num_flows', type=int, default=10,
                    metavar='NUM_FLOWS', help='Number of flow layers, ignored in absence of flows')

parser.add_argument('--z_size', type=int, default=64, metavar='ZSIZE',
                    help='how many stochastic hidden units')
parser.add_argument('--encoder_dim', type=int, default=256, metavar='ESIZE',
                    help='output feature dim of encoder')
parser.add_argument('--decoder_dim', type=int, default=256, metavar='DSIZE',
                    help='output feature dim of decoder') 

# gpu/cpu
parser.add_argument('--gpu_num', type=int, default=0, metavar='GPU', help='choose GPU to run on.')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

if args.manual_seed is None:
    args.manual_seed = random.randint(1, 100000)
random.seed(args.manual_seed)
torch.manual_seed(args.manual_seed)
np.random.seed(args.manual_seed)

if args.cuda:
    torch.cuda.set_device(args.gpu_num)


def run(args):

    print('\nMODEL SETTINGS: \n', args, '\n')
    print("Random Seed: ", args.manual_seed)


    train_loader, val_loader, test_loader, args = load_dataset(args)
    
    encoder = MLP_encoder(args)
    decoder = MLP_decoder(args)
    model = VAE.PlanarVAE(encoder, decoder, args)
    
    if args.cuda:
        print("Model on GPU")
        model.cuda()
        
    print(model)

    optimizer = optim.Adamax(model.parameters(), lr=args.learning_rate, eps=1.e-7)

    #### Training
    train_loss = []
    val_loss = []

    epoch = 0

    for epoch in range(1, args.epochs + 1):

        tr_loss = train(epoch, train_loader, model, optimizer, args)
        train_loss.append(tr_loss)

        v_loss = evaluate(val_loader, model, args)
        val_loss.append(v_loss)


    train_loss = np.hstack(train_loss)
    val_loss = np.array(val_loss)
    plot_training_curve(train_loss, val_loss)   
    
    #### Testing

    validation_loss = evaluate(val_loader, model, args)
    test_loss = evaluate(test_loader, model, args, testing=True)


  
if __name__ == "__main__":

    run(args)

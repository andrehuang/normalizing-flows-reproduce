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
#from util.plotting import plot_training_curve
import json
import pathlib
import time

parser = argparse.ArgumentParser(description='PyTorch Normalizing flows')

parser.add_argument('-d', '--dataset', type=str, default='mnist', choices=['mnist'],
                    metavar='DATASET',
                    help='Dataset choice.')

parser.add_argument('-nc', '--no_cuda', action='store_true', default=False,
                    help='disables CUDA training')

parser.add_argument('--manual_seed', type=int, help='manual seed, if not given resorts to random seed.')

parser.add_argument('-li', '--log_interval', type=int, default=10, metavar='LOG_INTERVAL',
                    help='how many batches to wait before logging training status')

parser.add_argument('-od', '--out_dir', type=str, default='logs/', metavar='OUT_DIR',
                    help='output directory for model snapshots etc.')

fp = parser.add_mutually_exclusive_group(required=False)
fp.add_argument('-te', '--testing', action='store_true', dest='testing',
                help='evaluate on test set after training')
fp.add_argument('-va', '--validation', action='store_false', dest='testing',
                help='only evaluate on validation set')
parser.set_defaults(testing=True)

# optimization settings
parser.add_argument('-e', '--epochs', type=int, default=1000, metavar='EPOCHS',
                    help='number of epochs to train (default: 1000)')
parser.add_argument('-es', '--early_stopping_epochs', type=int, default=100, metavar='EARLY_STOPPING',
                    help='number of early stopping epochs')

parser.add_argument('-bs', '--batch_size', type=int, default=100, metavar='BATCH_SIZE',
                    help='input batch size for training (default: 100)')
parser.add_argument('-lr', '--learning_rate', type=float, default=0.00001, metavar='LEARNING_RATE',
                    help='learning rate')

parser.add_argument('-a', '--anneal', type=str, default="std", choices= ["std", "off", "kl"], help="beta annealing scheme")
parser.add_argument('--max_beta', type=float, default=1., metavar='MB',
                    help='max beta for warm-up')
parser.add_argument('--min_beta', type=float, default=0.0, metavar='MB',
                    help='min beta for warm-up')
parser.add_argument('-f', '--flow', type=str, default='planar', choices=['planar', 'NICE', 'NICE_MLP', 'syl_orthogonal', 'real' ])
parser.add_argument('-nf', '--num_flows', type=int, default=10,
                    metavar='NUM_FLOWS', help='Number of flow layers, ignored in absence of flows')
parser.add_argument('-nov', '--num_orthonormal_vec', type=int, default=8, metavar='NUM_ORTHONORMAL_VEC',
                    help='For orthogonal flow: number of orthogonal vectors (M)')
parser.add_argument('--syl_ortho_invertible', type=bool, default=False, metavar='ORTHO_FLOW_INVERT',
                    help='select if we need the sylvester orthogonal flow to be invertible')
parser.add_argument('--z_size', type=int, default=40, metavar='ZSIZE',
                    help='how many stochastic hidden units')
parser.add_argument('--encoder_dim', type=int, default=400, metavar='ESIZE',
                    help='output feature dim of encoder')
parser.add_argument('--decoder_dim', type=int, default=400, metavar='DSIZE',
                    help='output feature dim of decoder') 

parser.add_argument('-vp', '--vampprior', type=bool, default=False, metavar='VAMPPRIOR',
                    help='choose whether to use VampPrior')
parser.add_argument('--num_pseudos', type=int, default=500, metavar='NUM_PSEUDOS',
                    help='number of pseudoinputs used for VampPrior')
parser.add_argument('--data_as_pseudo', type=bool, default=True, metavar='data_as_pseudo',
                    help='use random training data as pseudoinputs')

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
    if args.flow == "planar":
        model = VAE.PlanarVAE(encoder, decoder, args)
    elif args.flow == "NICE": # NICE-planar
        model = VAE.NICEVAE_amor(encoder, decoder, args)
    elif args.flow == "NICE_MLP":
        model = VAE.NICEVAE(encoder, decoder, args)
    elif args.flow == "syl_orthogonal":
        model = VAE.Sylvester_ortho_VAE(encoder, decoder, args)
    elif args.flow == "real":
        model = VAE.RealNVPVAE(encoder, decoder, args)
    

    
    if args.vampprior:
        load = torch.utils.data.DataLoader(train_loader.dataset, batch_size=args.num_pseudos, shuffle=True)
        pseudo_inputs = next(iter(load))[0] if args.data_as_pseudo else None
        model.init_pseudoinputs(pseudo_inputs)
    
    if args.cuda:
        print("Model on GPU")
        model.cuda()
        
    print(model)

    optimizer = optim.RMSprop(model.parameters(), lr=args.learning_rate, momentum=0.9)

    #### Training
    train_loss = []
    val_loss = []

    epoch = 0
    t = time.time()
    for epoch in range(1, args.epochs + 1):

        tr_loss = train(epoch, train_loader, model, optimizer, args)
        train_loss.append(tr_loss.mean())

        v_loss = evaluate(val_loader, model, args)
        val_loss.append(v_loss)


    train_loss = np.hstack(train_loss)
    val_loss = np.hstack(val_loss)
    #plot_training_curve(train_loss, val_loss)   
    results = {"train_loss": train_loss.tolist(), "val_loss": val_loss.tolist()}
    
    
    #### Testing

    validation_loss = evaluate(val_loader, model, args)
    test_loss, log_likelihood = evaluate(test_loader, model, args, testing=True)
    results["ELBO"] = test_loss
    results["log_likelihood"] = log_likelihood

    elapsed = time.time() - t
    results["Running time"] = elapsed

    # Save the results.
    json_dir = args.out_dir + f"{args.flow}perm_k_{args.num_flows}_RMSProp_lr{args.learning_rate}_4"
    print("Saving data at: " + json_dir)
    output_folder = pathlib.Path(json_dir)
    output_folder.mkdir(parents=True, exist_ok=True)
    results_json = json.dumps(results, indent=4, sort_keys=True)
    (output_folder / "results.json").write_text(results_json)

  
if __name__ == "__main__":

    run(args)

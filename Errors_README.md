# Errors

## Update March. 14
by Angeliki

For number of flows K=10, I encounter the following error during epoch 134:

RuntimeError: all elements of input should be between 0 and 1

To reproduce the error run main_experiment_VAE.py with K=10 and manual_seed = 55158:

%run main_experiment_VAE.py -nf 10 --manual_seed 55158



To find where the error comes from, I run the Sylvester reference code, with only using our own:

1. models.VAE , models.flows , models.model  
2. utils.load_data
3. data (folder)

Because of these changes, some modifications in the main_experiment.py should also be done 

- at line 85 add:

parser.add_argument('--encoder_dim', type=int, default=256, metavar='ESIZE',
                    help='output feature dim of encoder')
                    
parser.add_argument('--decoder_dim', type=int, default=256, metavar='DSIZE',
                    help='output feature dim of decoder')
                    
parser.add_argument('-db', '--dynamic_binarization', default=False) 

parser.add_argument('-it', '--input_type', default='binary') 

- at elif args.flow == 'planar' body at line 153 replace with:

    encoder = MLP_encoder(args)
    
    decoder = MLP_decoder(args)
    
    model = VAE.PlanarVAE(encoder, decoder, args)
    
- at the start of the file main_experiment.py, import:

from models.model import MLP_encoder, MLP_decoder


%run main_experiment_VAE.py -f 'planar' -nf 10 --manual_seed 55158

When running, I encountered the exact same error at epoch 134. Thus, I assume that VAE architecture/implementation is what causes this error.

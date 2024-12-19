import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   
lr = 1e-4
pad_length = 256
epochs = 10
batch_size = 32
dim_hidden = 256
use_pos_enc = False
use_rank_enc = False
n_layers = 6
model_type = "transformer"#, "mlp" "transformer"

config = {  
    'device': device,
    'lr': lr,
    'pad_length': pad_length,
    'epochs': epochs,
    'batch_size': batch_size,
    'dim_hidden': dim_hidden,
    'use_pos_enc': use_pos_enc,
    'use_rank_enc': use_rank_enc,
    'n_layers': n_layers,
    'model_type': model_type
}
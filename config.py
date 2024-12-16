import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   
lr = 1e-4
pad_length = 1024
epochs = 10
batch_size = 32
dim_hidden = 256
use_pos_enc = False
import torch
import torch.nn.functional as F
from config import *
import math
import torch.nn as nn   
class MLPModel(torch.nn.Module):
    def __init__(self):
        super(MLPModel, self).__init__()
        self.fc1 = torch.nn.Linear(768, 256)
        self.fc2 = torch.nn.Linear(256, 64)
        self.fc3 = torch.nn.Linear(64, 10)
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.softmax(x)
        return x

class EncoderModelPreTrain(nn.Module):
    #input should be (batchsize, num_pcas, dim_pcas)
    def __init__(self, num_classes, num_tokens, hidden_dim = dim_hidden,n_layers = n_layers):
        super().__init__()
        self.embedding = torch.nn.Embedding(num_tokens, hidden_dim)
        self.module_list = nn.ModuleList([nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=hidden_dim//64,dim_feedforward=4*hidden_dim, batch_first=True, activation='gelu') for i in range(n_layers)])
        self.classification_token = nn.Parameter(torch.Tensor(hidden_dim), requires_grad=True)
        nn.init.uniform_(self.classification_token, a=-1/math.sqrt(hidden_dim), b=1/math.sqrt(hidden_dim))
        self.dense = nn.Linear(hidden_dim, num_classes)

        if use_pos_enc:
            self.positional_encoding = self.generate_positional_encoding(pad_length, dim_hidden).to(device)
        if use_rank_enc:
            self.rank = torch.arange(0, pad_length, dtype=torch.float).unsqueeze(1).to(device)
            self.rank_encoding = nn.Sequential(nn.Linear(1, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))

    
    def generate_positional_encoding(self, pad_length, dim_hidden):
        pe = torch.zeros(pad_length, dim_hidden)
        position = torch.arange(0, pad_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim_hidden, 2).float() * (-math.log(10000.0) / dim_hidden))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        return pe

    def forward(self, x, mask):
        x = self.embedding(x)
        if use_rank_enc:
            x = x * self.rank_encoding(self.rank)
        if use_pos_enc:
            x = x + self.positional_encoding

        
        classification_token = torch.stack([self.classification_token.unsqueeze(0) for _ in range(x.shape[0])])
        x = torch.cat((classification_token,x),dim = 1)

        #also add one token to the mask
        mask = torch.cat((torch.zeros(mask.shape[0],1).bool().to(mask.device),mask),dim = 1)

        for layer in self.module_list:
            x = layer(x, src_key_padding_mask=mask)
    
        classification_token = x[:,0,:]
        return self.dense(classification_token)
    

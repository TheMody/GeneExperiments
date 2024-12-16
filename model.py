import torch
import torch.nn.functional as F
from config import *
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
    
class TransformerModel(torch.nn.Module):
    def __init__(self, num_tokens, num_classes):
        super(TransformerModel, self).__init__()
        self.embedding = torch.nn.Embedding(num_tokens, dim_hidden)
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=dim_hidden, nhead=4)
        self.transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers=6, enable_nested_tensor= False, mask_check=False)
        self.fc = torch.nn.Linear(dim_hidden, num_classes)
        #positional encoding for transformers
        if use_pos_enc:
            self.positional_encoding = torch.nn.Parameter(torch.randn(1, pad_length, dim_hidden))

    def forward(self, x):
        x = self.embedding(x)
        if use_pos_enc:
            x = x + self.positional_encoding
        x = self.transformer(x)
        x = self.fc(torch.mean(x, dim=1))
        return x


import torch
import numpy as np
from datasets import load_from_disk
from dataset_info import *
from config import *
from tqdm import tqdm
class Dataset(torch.utils.data.Dataset):   
    #returns the cell type and the input_ids as integer values
    def __init__(self):
        dataset_path = 'celltype_dataset.dataset'
        self.dataset = load_from_disk(dataset_path)



    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        x = self.dataset[idx]["input_ids"] 
     #   print(len(x))
        mask = torch.zeros(pad_length)
        if len(x) < pad_length:
            mask[len(x):] = 1
            x = np.pad(x, (0, pad_length-len(x)), 'constant') 
        else:
            x = x[:pad_length]
        x = torch.tensor(x), mask.to(torch.bool)
        y = self.dataset[idx]["cell_type"]
        y = torch.tensor(cell_types.index(y))
        return x,y
    
if __name__ == "__main__":
    ds = Dataset()
    for i in tqdm(range(len(ds))):
       print(ds[i][0])
       if (ds[i][0]).dtype != torch.int64:
           print("problem with input_ids at position: ", i, (ds[i][0]).dtype)

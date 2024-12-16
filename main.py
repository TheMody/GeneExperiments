#import os
#os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


import torch
import numpy as np
from model import MLPModel, TransformerModel
from dataset_info import *
from config import *
from dataset import Dataset
import wandb
from cosine_scheduler import CosineWarmupScheduler
import time
from tqdm import tqdm
def train():

    model = TransformerModel(num_tokens=Num_ids, num_classes=len(cell_types))
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    dataset = Dataset()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    scheduler = CosineWarmupScheduler(optimizer, warmup=100, max_iters=len(dataset)//batch_size *epochs)
    wandb.init(project="celltype_classification", config=config)
    loss_avg = 0.0
    accuracy_avg = 0.0
    with tqdm(total=len(dataset)//batch_size *epochs) as pbar:
        for epoch in range(epochs):
            for i, (x, y) in enumerate(dataloader):
                start = time.time()
                x = x.to(device)
                y = y.to(device)

                if torch.max(x).item() >= Num_ids:
                    print("Invalid cell type", x)
                    print(torch.max(x).item())
                    continue

                optimizer.zero_grad()
                y_pred = model(x)
                loss = criterion(y_pred, y)
                loss.backward()
                optimizer.step()
                scheduler.step()

                log_dict = {}
                #if i % 100 == 0:
                   # print(f"Epoch: {epoch}, Iteration: {i}, Loss: {loss.item()}")
                accuracy = y_pred.argmax(dim=1).eq(y).sum().item() / len(y)
                log_dict["accuracy"] = accuracy
                log_dict["loss"] = loss.item()
                log_dict["time_per_step"] = time.time() - start
                log_dict["lr"] = optimizer.param_groups[0]["lr"]

                wandb.log(log_dict)
                loss_avg = 0.99 * loss_avg + 0.01 * loss.item()
                accuracy_avg = 0.99 * accuracy_avg + 0.01 * accuracy
                pbar.set_description(f"Loss: {loss_avg}, Accuracy: {accuracy_avg}")
                pbar.update(1)

if __name__ == "__main__":
    train()



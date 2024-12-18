#import os
#os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


import torch
import numpy as np
from model import MLPModel,EncoderModelPreTrain
from dataset_info import *
from config import *
from dataset import Dataset
import wandb
from cosine_scheduler import CosineWarmupScheduler
import time
from tqdm import tqdm
def train():

    model = EncoderModelPreTrain(num_classes=len(cell_types), num_tokens=Num_ids)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    dataset = Dataset()
    split = int(0.9 * len(dataset))
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [split, len(dataset) - split])
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    scheduler = CosineWarmupScheduler(optimizer, warmup=100, max_iters=len(dataset)//batch_size *epochs)
    wandb.init(project="celltype_classification", config=config)
    loss_avg = 0.0
    accuracy_avg = 0.0
    with tqdm(total=len(dataset)//batch_size *epochs) as pbar:
        model.train()
        for epoch in range(epochs):
            for i, (x, y) in enumerate(dataloader):
                start = time.time()
                x, mask = x
                mask = mask.to(device)
                x = x.to(device)
                y = y.to(device)

                #print(mask)

                if torch.max(x).item() >= Num_ids:
                    print("Invalid cell type", x)
                    print(torch.max(x).item())
                    continue

                optimizer.zero_grad()
                y_pred = model(x, mask)
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

            val_loss = 0.0
            val_acc = 0.0
            model.eval()
            for i, (x, y) in enumerate(val_dataloader):
                x, mask = x
                mask = mask.to(device)
                x = x.to(device)
                y = y.to(device)
                y_pred = model(x, mask)
                accuracy = y_pred.argmax(dim=1).eq(y).sum() / len(y)
                loss = criterion(y_pred, y)
                val_loss += loss.item()
                val_acc += accuracy.item()
            val_loss /= len(val_dataloader)
            val_acc /= len(val_dataloader)
            wandb.log({"val_loss": val_loss, "val_accuracy": val_acc})
            print(f"Validation Loss: {val_loss}, Validation Accuracy: {val_acc}")

if __name__ == "__main__":
    train()



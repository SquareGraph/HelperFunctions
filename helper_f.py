import numpy as np
import pandas as pd
import torchmetrics
import requests
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision
from tqdm import tqdm
from typing import Tuple, Dict, List
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"

# PyTorch Train,test & eval loop
# ______________________________________________________________________
# ________________________EVALUATION MODE_______________________________
# ______________________________________________________________________

def eval_model(model: torch.nn.Module, #we want to ensure dtype here
               data_loader: torch.utils.data.DataLoader, #data will be fed through the PyTorch DataLoaders
               loss_fn: torch.nn.Module, #ensuring dtype of loss function
               accuracy_fn, device:torch.device=device): #device agnostic code
    """ Return a dictionary containing the results of model prediction on data_loader """

    loss, acc = 0,0 #starting values for each of the params
    preds = []
    model.eval() #turning evaluation mode (we don't need to calc grad here)
    with torch.inference_mode(): #context manager for above
        for X,y in tqdm(data_loader): #evaluation for loop with tqdm progress bar

            X,y = X.to(device), y.to(device) #sending X and y to the same device as model, preferable cuda

            y_preds = model(X) #feeding X's to the model
            preds.append(y_preds)

            loss += loss_fn(y_preds, y) #calculate and accumulate loss
            acc += accuracy_fn(y_preds.argmax(dim=1), y) #same for accuracy

        loss /= len(data_loader) #outside of a for loop let's devide accumulated stats by a length of a datastream to get the mean.
        acc /= len(data_loader) # same.

    return {"model_name": model.__class__.__name__,
            "model_loss": loss.item(),
            "model_acc": acc.item(),
            "model_preds":preds} #return dictionary of loss and accuracy of a model.

# ______________________________________________________________________
# ________________________TRAIN STEP_______________________________
# ______________________________________________________________________

# I'll comment only if there is something new.
def train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn_: torch.nn.Module,
               optimizer_obj: torch.optim.Optimizer,  # check the dtype for the optimizer
               accuracy_fn, device: torch.device = device):
    """train model for an epochs"""
    # training

    train_loss, train_acc = 0, 0
    model.train()

    # Training loop. Enumerate function assign count to the "batch" variable, and X&y comes from the dataloader itself.
    for batch, (X, y) in enumerate(data_loader):
        X, y = X.to(device), y.to(device)  #

        # 1. Forward Pass
        preds = model(X)

        # 2. Calculate the loss
        loss = loss_fn_(preds, y)

        train_loss += loss
        train_acc += accuracy_fn(preds.argmax(dim=1), y)

        # 3. Zero Grad
        optimizer_obj.zero_grad()  # we have to zero gradient, otherwise it will accumulate

        # 4. Loss backwards propagation
        loss.backward()

        # 5. Optimizer step
        optimizer_obj.step()

        # Divide total train_loss by the size of a train data to get the mean.
    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    print(
        f"\nCurrent training loss: {train_loss}\nCurrent training acc: {train_acc}%\n")  # here is an issue... not showing results

# ______________________________________________________________________
# ________________________TEST STEP_______________________________
# ______________________________________________________________________


def test_step(model: torch.nn.Module,
              data_loader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              accuracy_fn, device: torch.device = device):
    """ Return a dictionary containing the results of model prediction on data_loader """

    # set up test acc and loss
    test_loss, test_acc = 0, 0

    # turn on evaluation mode
    model.eval()

    # turn on context manager for testing
    with torch.inference_mode():
        # set up testing loop
        for X, y in tqdm(data_loader):
            # send X and y to target device
            X, y = X.to(device), y.to(device)

            # 1. Froward Pass
            y_preds = model(X)

            # 2. Accumulate stats
            test_loss += loss_fn(y_preds, y)
            test_acc += accuracy_fn(y_preds.argmax(dim=1), y)

            # 3. Calculate actual parameters
        test_loss /= len(data_loader)
        test_acc /= len(data_loader)
        print(f"\nCurrent test loss: {test_loss}\nCurrent test acc: {test_acc}%\n")

    return {"model_name": model.__class__.__name__,
            "model_loss": test_loss.item(),
            "model_acc": test_acc}


# ______________________________________________________________________
# ________________________PLOT RANDOM IMAGES_______________________________
# ______________________________________________________________________

def plot_ranom_images(start_index: int,
                       end_index: int,
                       sample_size: int,
                      X: torch.Tensor, y: torch.Tensor,
                      figsize: Tuple,
                      rowsncols: Tuple,
                      font_size: int):

    """Plot random samples of data from dataset in format Channel/Height/Width"""
    random_idx = np.random.randint(start_index,
                                   end_index,
                                   sample_size) #list of indexes for sample games

    test_samples = []
    test_labels = []

    for i in random_idx:
        test_samples.append(X[i].permute(1,2,0))
        test_labels.append(y[i].item())

    plt.figure(figsize=(figsize[0],figsize[1]))

    nrows = rowsncols[0]
    ncols = rowsncols[1]

    for i, sample in enumerate(test_samples):
        plt.subplot(nrows,ncols, i+1)
        plt.imshow(sample.squeeze())
        plt.axis(False)
        title_label = y[i].item()
        sample_index = random_idx[i]
    plt.title((title_label,sample_index), fontsize=font_size)

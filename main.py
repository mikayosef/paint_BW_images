import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, dataloader
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

# from main import main
from BW_class import BWDataset
from net import CONVNet
from net import FULLNet
from train_val import train
from train_val import val
from preview import imshow
from preview import preview


def main():
    torch.backends.cudnn.benchmark = True

    # main
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 2

    trainset = BWDataset(train=True, download=True, device=device)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=0
    )

    valset = BWDataset(train=False, download=True, device=device)
    valloader = torch.utils.data.DataLoader(
        valset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    torch.cuda.is_available()

    NET = CONVNet().to(device)

    criterion = nn.L1Loss(reduction="mean").to(device)
    optimizer = optim.Adam(NET.parameters(), lr=0.0001)

    train_set_loss, val_set_loss, number_of_epochs = train(
        2, trainloader, valloader, optimizer, criterion, NET
    )

    PATH = "./data/conv_2epoch.ckpt"
    torch.save(NET.state_dict(), PATH)  # conv model, 2 epochs

    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.plot(number_of_epochs, train_set_loss, "r", label="Train")
    plt.plot(number_of_epochs, val_set_loss, "b", label="Validate")
    plt.xlabel("Number_of_epochs")
    plt.ylabel("Loss")
    plt.legend()

    plt.show()


if __name__ == "__main__":
    main()

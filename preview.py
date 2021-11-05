from torch.utils.data import Dataset, DataLoader, dataloader
import torchvision
import matplotlib.pyplot as plt


import numpy as np


def imshow(img):
    npimg = img
    # npimg = np.clip(npimg, 0, 1)
    plt.imshow(npimg)
    plt.show()


def imshow_to_numpy(img):
    npimg = img.cpu().numpy()
    npimg = np.clip(npimg, 0, 1)
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def preview(trainloader):

    # # # get some random training images
    dataiter = iter(trainloader)
    gray, col = dataiter.next()
    # show gray images
    imshow_to_numpy(torchvision.utils.make_grid(gray))
    plt.figure()
    # print colored
    imshow_to_numpy(torchvision.utils.make_grid(col))

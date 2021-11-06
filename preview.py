import torchvision
import matplotlib.pyplot as plt
import numpy as np


def imshow(img):
    npimg = img
    if img.shape[0] == 1 or len(img.shape) == 2:
        plt.imshow(npimg, cmap="gray")
    else: plt.imshow(npimg)
    plt.show()


def imshow_to_numpy(img):
    npimg = img.cpu().numpy()
    npimg = np.clip(npimg, 0, 1)
    npimg = np.transpose(npimg, (1, 2, 0))
    plt.imshow(npimg)
    plt.show()

def preview(trainloader):   
    dataiter = iter(trainloader)
    gray, col = dataiter.next()

    imshow_to_numpy(torchvision.utils.make_grid(gray))
    plt.figure()
    imshow_to_numpy(torchvision.utils.make_grid(col))

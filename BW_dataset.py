from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms


class BWDataset(Dataset):
    def __init__(self, train, download, device):
        self.data = torchvision.datasets.CIFAR10(
            root="./data", train=train, download=download)
        self.device = device

    def __len__(self):
        return len(self.data
        )

    def __getitem__(self, idx):
        img, label = self.data[idx]
        img_transform = transforms.Compose([transforms.ToTensor()])
        img = img_transform(img)
        gray_transform = torchvision.transforms.Grayscale(num_output_channels=1)
        gray = gray_transform(img)
        return gray, img



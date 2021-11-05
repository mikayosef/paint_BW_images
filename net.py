import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision.transforms.transforms import ToTensor


class FULLNet(nn.Module):
    def __init__(self):
        super().__init__()
        # nn.Sequential()
        self.conv1 = nn.Conv2d(1, 30, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(30, 15, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(15, 8, kernel_size=3, padding=1)
        # self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(8 * 32 * 32, 3 * 32 * 32)
        # self.fc2 = nn.Linear(5 * 32 * 32, 4 * 32 * 32)
        # self.fc3 = nn.Linear(5 * 32 * 32, 3 * 32 * 32)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = self.fc3(x)

        batch_size, vector_dim = x.shape
        # batch_size = torch.ByteTensor(batch_size)
        x = torch.reshape(x, (batch_size, 3, 32, 32))
        # x = torch.reshape(x, [3, 32, 32]).transpose(0, 1, 2)
        return x




class CONVNet(nn.Module):
    def __init__(self):
        super().__init__()
        # nn.Sequential()
        self.conv1 = nn.Conv2d(1, 30, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(30, 60, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(60, 100, kernel_size=3, padding=1)

        self.conv4 = nn.Conv2d(100, 150, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(150, 70, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(70, 15, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(15, 3, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        return x
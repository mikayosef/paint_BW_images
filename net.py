import torch.nn as nn
import torch.nn.functional as F
import torch


class CONVNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 12, kernel_size=3,
                               padding=1, padding_mode='reflect')
        self.conv2 = nn.Conv2d(12, 25, kernel_size=3,
                               padding=1, padding_mode='reflect')
        self.conv3 = nn.Conv2d(25, 40, kernel_size=3,
                               padding=1, padding_mode='reflect')
        self.conv4 = nn.Conv2d(40, 55, kernel_size=3,
                               padding=1, padding_mode='reflect')
        self.conv5 = nn.Conv2d(55, 70, kernel_size=3,
                               padding=1, padding_mode='reflect')
        self.conv6 = nn.Conv2d(70, 60, kernel_size=3,
                               padding=1, padding_mode='reflect')
        self.conv7 = nn.Conv2d(60, 40, kernel_size=3,
                               padding=1, padding_mode='reflect')
        self.conv8 = nn.Conv2d(40, 20, kernel_size=3,
                               padding=1, padding_mode='reflect')
        self.conv9 = nn.Conv2d(20, 10, kernel_size=3,
                               padding=1, padding_mode='reflect')
        self.conv10 = nn.Conv2d(10, 5, kernel_size=3,
                                padding=1, padding_mode='reflect')
        self.conv11 = nn.Conv2d(5, 3, kernel_size=3,
                                padding=1, padding_mode='reflect')

        # self.conv = []
        # input_dim = 3
        # output_dim = 15
        # for i in range(10):
        #     self.conv.append(nn.Sequential(nn.Conv2d(input_dim, output_dim, kernel_size=3, padding=1), nn.ReLU()))
        #     input_dim = output_dim
        #     output_dim = int(output_dim*1.1)
        # self.conv = nn.Sequential(*self.conv)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = F.relu(self.conv9(x))
        x = F.relu(self.conv10(x))
        x = F.relu(self.conv11(x))

        return x


# small network with convolution and FC:

class FULLNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 30, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(30, 15, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(15, 8, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(8 * 32 * 32, 3 * 32 * 32)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))

        batch_size, vector_dim = x.shape
        x = torch.reshape(x, (batch_size, 3, 32, 32))
        return x

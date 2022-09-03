from Net_128_4conv import Net
from torch import nn


class LinearNet(nn.Module):
    def __init__(self):
        super(LinearNet, self).__init__()
        self.ConvNet = Net()
        self.fc = nn.Sequential(
            nn.Linear(2 * 128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, img1, img2):
        vector1 = self.ConvNet(img1)
        vector2 = self.ConvNet(img2) if img1.shape == img2.shape else img2
        return vector1, vector2

from torch import nn


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 6, 7),
            nn.Tanh(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 6),
            nn.Tanh(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 5),
            nn.Tanh(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 48, 5),
            nn.Tanh(),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(48 * 4 * 4, 512),
            nn.ReLU(),
            # more linear
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

    def forward(self, img):
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0], -1))
        return output

import torch

from torch import nn

class PrunedConvNetIDS(nn.Module):
    def __init__(self):
        super(PrunedConvNetIDS, self).__init__()

        self.feature_extraction_layer = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=27, kernel_size=5, stride=1, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(27, eps=0.001, momentum=0.9),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=27, out_channels=26, kernel_size=5, stride=1, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(26, eps=0.001, momentum=0.9),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.binary_classification_layer = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features=8294, out_features=64),
            nn.Dropout(p=0.3),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=1)
        )

    def forward(self, x):
        x = self.feature_extraction_layer(x)
        x = torch.flatten(x, 1)
        x = self.binary_classification_layer(x)
        x = torch.sigmoid(x)
        return x

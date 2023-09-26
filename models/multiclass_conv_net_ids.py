import torch

from torch import nn

class MultiClassConvNetIDS(nn.Module):
    def __init__(self, number_of_outputs = 2):
        super(MultiClassConvNetIDS, self).__init__()

        self.feature_extraction_layer = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(32, eps=0.001, momentum=0.9),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(64, eps=0.001, momentum=0.9),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.binary_classification_layer = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features=20416, out_features=64),
            nn.Dropout(p=0.3),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=number_of_outputs)
        )

    def forward(self, x):
        x = self.feature_extraction_layer(x)
        x = torch.flatten(x, 1)
        x = self.binary_classification_layer(x)
        x = torch.sigmoid(x)
        return x

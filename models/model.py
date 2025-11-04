import torch
from torch import nn

# Define the custom neural network
class CustomNet(nn.Module):
    def __init__(self):
        super(CustomNet, self).__init__()
        # Define layers of the neural network

        # self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, stride=1)  # <-- (64, 224, 224)
        # self.relu1 = nn.ReLU()
        # self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)          # <-- (128, 224, 224)
        # self.relu2 = nn.ReLU()
        # self.mp1 = nn.MaxPool2d(2, 2)

        # self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=1)
        # self.relu3 = nn.ReLU()
        # self.mp2 = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, stride=1),     # <-- (64, 224, 224)
            nn.BatchNorm2d(64),   # si applicano in ordine prima batchnorm e poi relu
            nn.ReLU(),            # se si facesse il contrario si introdurrebbe distorsione
            nn.MaxPool2d(2,2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),             # <-- (128, 112, 112)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=1),  # <-- (256, 56, 56)
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )

        self.aap = nn.AdaptiveAvgPool2d((1,1))

        self.fc1 = nn.Linear(256, 200) # 200 is the number of classes in TinyImageNety


    def forward(self, x):
        # Define forward pass

        # print("Shape before conv1:", x.shape)

        x = self.conv1(x)

        # print("Shape after conv1:", x.shape)

        x = self.conv2(x)

        # print("Shape after conv2:", x.shape)

        x = self.conv3(x)

        # print("Shape after conv3:", x.shape)

        x = self.aap(x)

        # print("Shape afeter aap:", x.shape)

        x = x.flatten(start_dim=1)

        # print("Shape afer flatteninig:", x.shape)

        x = self.fc1(x)

        # print("Shape after fc1:", x.shape)

        return x
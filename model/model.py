import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class BaseModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)

        x = self.avgpool(x)
        x = x.view(-1, 128)
        return self.fc(x)


class ResNet50_ft(nn.Module):

    def __init__(self, num_classes):
        super().__init__()

        self.model = models.resnet50(pretrained=True) # pre-trained model로 resnet50 지정
        num_ftrs = self.model.fc.in_features  # resnet의 원래 출력layer수 (2048)
        self.model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.7),
            nn.Linear(512, num_classes))

        # nn.init.xavier_uniform_(self.model.fc.weight) #xavier_unifrom 적용
        # self.stdv = 1. / math.sqrt(self.model.fc.weight.size(1))
        # self.model.fc.bias.data.uniform_(-self.stdv, self.stdv)

        self.ct = 0
        for child in self.model.children():
            self.ct += 1
            if self.ct < 6:
                for param in child.parameters():
                    param.requires_grad = False

    def forward(self, x):
        return self.model(x)
import torch
import torch.nn as nn
import torch.nn.functional as F


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


# Custom Model Template
class MyModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        """
        1. 위와 같이 생성자의 parameter 에 num_claases 를 포함해주세요.
        2. 나만의 모델 아키텍쳐를 디자인 해봅니다.
        3. 모델의 output_dimension 은 num_classes 로 설정해주세요.
        """
        

    def forward(self, x):
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """
        return x

# ResNet18
class ResNet18(nn.Module):

    def __init__(self, num_classes):
        super().__init__()

        from torchvision.models import resnet18
        self.model = resnet18(pretrained=True)
        self.model.fc = nn.Linear(512, num_classes)
        import math
        nn.init.xavier_uniform_(self.model.fc.weight)
        self.stdv = 1. / math.sqrt(self.model.fc.weight.size(1))
        self.model.fc.bias.data.uniform_(-self.stdv, self.stdv)
        # print("=" * 100)
        # print(self.model)
        # print("=" * 100)  

    def forward(self, x):
        return self.model(x)

# ResNet18 - fine tuning
class ResNet18_ft(nn.Module):

    def __init__(self, num_classes):
        super().__init__()

        from torchvision.models import resnet18
        self.model = resnet18(pretrained=True)
        self.model.fc = nn.Linear(512, num_classes)
        import math
        nn.init.xavier_uniform_(self.model.fc.weight)
        self.stdv = 1. / math.sqrt(self.model.fc.weight.size(1))
        self.model.fc.bias.data.uniform_(-self.stdv, self.stdv)

        self.ct = 0
        for child in self.model.children():
            self.ct += 1
            if self.ct < 6:
                for param in child.parameters():
                    param.requires_grad = False
        # print("=" * 100)
        # print(self.model)
        # print("=" * 100)  

    def forward(self, x):
        return self.model(x)


# DenseNet 
class densenet(nn.Module):

    def __init__(self, num_classes=18):
        super().__init__()

        from torchvision.models import densenet161
        self.model = densenet161(pretrained=True)
        self.model.classifier = nn.Linear(2208, num_classes)
        # import torchvision.models as models
        # self.model = models.efficientnet_b5(pretrained=True)

        # import math
        # nn.init.xavier_uniform_(self.model.fc.weight)
        # self.stdv = 1. / math.sqrt(self.model.fc.weight.size(1))
        # self.model.fc.bias.data.uniform_(-self.stdv, self.stdv)

        # self.ct = 0
        # for child in self.model.children():
        #     self.ct += 1
        #     if self.ct < 6:
        #         for param in child.parameters():
        #             param.requires_grad = False
        print("=" * 100)
        print(self.model)
        print("=" * 100)  

    def forward(self, x):
        return self.model(x)
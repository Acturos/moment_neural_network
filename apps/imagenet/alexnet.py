# -*- coding: utf-8 -*-
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from mnn_core.mnn_pytorch import *
from torch.utils.tensorboard import SummaryWriter

import json
"""
AlexNet(
  (features): Sequential(
    (0): Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
    (1): ReLU(inplace=True)
    (2): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
    (3): Conv2d(64, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
    (4): ReLU(inplace=True)
    (5): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
    (6): Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (7): ReLU(inplace=True)
    (8): Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (9): ReLU(inplace=True)
    (10): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (11): ReLU(inplace=True)
    (12): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (avgpool): AdaptiveAvgPool2d(output_size=(6, 6))
  (classifier): Sequential(
    (0): Dropout(p=0.5, inplace=False)
    (1): Linear(in_features=9216, out_features=4096, bias=True)
    (2): ReLU(inplace=True)
    (3): Dropout(p=0.5, inplace=False)
    (4): Linear(in_features=4096, out_features=4096, bias=True)
    (5): ReLU(inplace=True)
    (6): Linear(in_features=4096, out_features=1000, bias=True)
  )
)

"""


class AlexNet(nn.Module):
    def __init__(self, num_classes: int = 1000) -> None:
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class Mnn_Alex_Net(nn.Module):
    def __init__(self, num_classes: int = 1000) -> None:
        super(Mnn_Alex_Net, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.fc1 = Mnn_Linear_without_Corr(256 * 6 * 6, 4096, bias=True)
        self.fc2 = Mnn_Linear_without_Corr(4096, 4096, bias=True)
        self.fc3 = Mnn_Linear_without_Corr(4096, num_classes, bias=True)
        self.a1 = Mnn_Activate_Mean.apply
        self.a2 = Mnn_Activate_Std.apply
        self.dropout = nn.Dropout()
        self.add_noise = 0.0
        self.mul_noise = 1.0
        self.scaling = 100

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        y = torch.sqrt(torch.abs(x)) * self.mul_noise + self.add_noise
        x, y = self.fc1(x, y)
        x_a = self.a1(x, y)
        y_a = self.a2(x, y, x_a)
        x_a *= self.scaling
        y_a *= self.scaling
        temp = torch.zeros_like(y_a)
        x = self.dropout(x_a)
        y = torch.where(x == 0, temp, y_a)
        x, y = self.fc2(x, y)
        x_a = self.a1(x, y)
        y_a = self.a2(x, y, x_a)
        x_a *= self.scaling
        y_a *= self.scaling
        x, y = self.fc3(x_a, y_a)
        return x


class Check_ImageNet_Model:
    def __init__(self):
        self.classes = None
        self.BATCH = 10
        self.LR = 0.1
        self.data_path = 'D:\Data_repos\Imagenet2012'
        self.get_imagenet_classes()
        
    def data_preprocess(self, split="val"):
        data_path = self.data_path
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        val_set = torchvision.datasets.ImageNet(
            root=data_path,
            split=split,
            transform=preprocess
        )
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=self.BATCH, shuffle=True)
        return val_set, val_loader

    def test_raw(self, model, data_loader, stop=500):
        correct = 0
        total = 0
        with torch.no_grad():
            for test_x, test_y in data_loader:
                test_x = test_x.type(torch.float64)
                output = model(test_x)
                _, predicted = torch.max(output.data, 1)
                total += test_y.size(0)
                correct += (predicted == test_y).sum().item()
                if total >= stop:
                    break
    
        accuracy = 100 * correct / total
        print("total samples: {:}, num of correct: {:}".format(total, correct))
        print('Top-1 Accuracy of the network is: %.4f %%' % accuracy)

    def get_imagenet_classes(self, file_path="imagenet_class_index.json"):
        file = open(file_path, 'r')
        load_dict = json.load(file)
        self.classes = list()
        for i in range(1000):
            self.classes.append(load_dict[str(i)][1])


if __name__ == "__main__":
    mnn_core_func.t_ref = 0.0
    model = torchvision.models.alexnet(pretrained=True)
    model.eval()
    check = Mnn_Alex_Net()
    check.load_state_dict(model.state_dict(), strict=False)
    check.scaling = 1.0
    test = Check_ImageNet_Model()
    dataset, dataloader = test.data_preprocess()
    #test.test_raw(model, dataloader)
    test.test_raw(check, dataloader)

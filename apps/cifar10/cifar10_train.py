# -*- coding: utf-8 -*-
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from mnn_core.mnn_modules import *
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/cifar10_experiment_1')


class Mnn_Conv2d_Compose_without_Rho(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, need_pool=True):
        super(Mnn_Conv2d_Compose_without_Rho, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.need_pool = need_pool

        self.mean_conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.std_conv2d = Mnn_Std_Conv2d_without_Rho(out_channels)

        self.mean_bn2d = nn.BatchNorm2d(out_channels)
        self.mean_bn2d.weight.data.fill_(2.5)
        self.mean_bn2d.bias.data.fill_(2.5)

        self.std_bn2d = Mnn_Std_Bn2d(out_channels)
        if need_pool:
            self.avg_pool2d = Mnn_AvgPool2d_without_Rho(kernel_size=2, stride=2)

    def forward(self, mean, std):
        mean = self.mean_conv2d(mean)
        std = self.std_conv2d(self.mean_conv2d, std)

        uhat = self.mean_bn2d(mean)
        shat = self.std_bn2d(self.mean_bn2d, mean, std)

        u = Mnn_Activate_Mean.apply(uhat, shat)
        s = Mnn_Activate_Std.apply(uhat, shat, u)

        if self.need_pool:
            u, s = self.avg_pool2d(u, s)
        return u, s


class Mnn_Alex_Like(nn.Module):
    def __init__(self, num_classes=10):
        super(Mnn_Alex_Like, self).__init__()
        self.conv1 = Mnn_Conv2d_Compose_without_Rho(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = Mnn_Conv2d_Compose_without_Rho(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = Mnn_Conv2d_Compose_without_Rho(32, 64, kernel_size=3, stride=1, padding=1)

        self.linear1 = Mnn_Layer_without_Rho(64*4*4, 64*4)
        self.linear2 = Mnn_Layer_without_Rho(64*4, 64*4)
        self.linear3 = Mnn_Linear_without_Corr(64*4, num_classes)

    def forward(self, ubar, sbar):
        ubar, sbar = self.conv1(ubar, sbar)
        ubar, sbar = self.conv2(ubar, sbar)
        ubar, sbar = self.conv3(ubar, sbar)

        ubar = ubar.view(ubar.size(0), 64*4*4)
        sbar = sbar.view(sbar.size(0), 64*4*4)

        ubar, sbar = self.linear1(ubar, sbar)
        ubar, sbar = self.linear2(ubar, sbar)
        uabr, sbar = self.linear3(ubar, sbar)
        return ubar, sbar


class Train_Cifar10_Model:
    def __init__(self):
        self.BATCH_SIZE = 512
        self.data_path = 'D:\Data_repos\Cifar10'
        self.EPOCHS = 15
        self.seed = 1024
        self.LR = 0.01
        self.classes = ('plane', 'car', 'bird', 'cat',
                   'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        self.train_loader = None
        self.test_loader = None
        self.fetch_data()

    def fetch_data(self):
        trainset = torchvision.datasets.CIFAR10(
            root=self.data_path,
            train=True,
            download=False,
            transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomGrayscale(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
            ])
        )

        self.train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=self.BATCH_SIZE, shuffle=True)

        testset = torchvision.datasets.CIFAR10(
            root=self.data_path,
            train=False,
            download=False,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
            ])
        )

        self.test_loader = torch.utils.data.DataLoader(
            testset, batch_size=self.BATCH_SIZE, shuffle=True)

    def train_alex(self, fix_seed=True, save_op=True, save_name="cifar_alex.pt", log_interval=10):
        if fix_seed:
            torch.manual_seed(self.seed)
        net = Mnn_Alex_Like()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(net.parameters(), lr=self.LR)
        net.train()
        count = 0
        print("------- CIFAR10 AlexNet Training Start-------")
        for epoch in range(self.EPOCHS):
            running_loss = 0
            for step, (batch_x, batch_y) in enumerate(self.train_loader):
                optimizer.zero_grad()
                batch_x = batch_x.type(torch.float64)
                output = net(batch_x, torch.sqrt(torch.abs(batch_x)))[0]
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                if step % log_interval == (log_interval - 1):
                    print('[%d, %5d] loss: %.4f' %
                          (epoch + 1, step + 1, running_loss / log_interval))
                    writer.add_scalar("loss/" + save_name[0:-3], running_loss / log_interval, count)
                    running_loss = 0.0
                    correct = 0
                    total = 0
                    _, predict = torch.max(output.data, 1)
                    total += batch_y.size(0)
                    correct += (predict == batch_y).sum().item()
                    writer.add_scalar("accuracy/" + save_name[0:-3], 100.0 * correct / total, count)
                    count += 1
                    print('Accuracy of the network on the %d tran images: %.3f %%' % (total, 100.0 * correct / total))
        print('----- Train Finished -----')
        if save_op:
            torch.save(net, save_name)
        net.eval()
        print('------ Test {:} Start -----'.format(save_name[0:-3]))
        correct = 0
        total = 0
        with torch.no_grad():
            for test_x, test_y in self.test_loader:
                images, labels = test_x, test_y
                images = images.type(torch.float64)
                sbar = torch.sqrt(torch.abs(images.clone()))
                output = net(images, sbar)[0]
                _, predicted = torch.max(output.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print("total samples: {:}, num of correct: {:}".format(total, correct))
        print('Accuracy of the network is: %.4f %%' % accuracy)


if __name__ == "__main__":
    test = Train_Cifar10_Model()
    test.train_alex()


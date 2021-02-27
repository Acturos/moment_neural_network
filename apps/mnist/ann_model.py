# -*- coding: utf-8 -*-
import torch
import torchvision
import pickle


class MLP_Sigmoid_Bn(torch.nn.Module):
    def __init__(self):
        super(MLP_Sigmoid_Bn, self).__init__()
        self.layer1 = torch.nn.Linear(784, 800, bias=False)
        self.bn = torch.nn.BatchNorm1d(800)
        self.bn.weight.data.fill_(2.5)
        self.bn.bias.data.fill_(2.5)

        self.sigmoid = torch.nn.Sigmoid()
        self.layer2 = torch.nn.Linear(800, 10)

    def forward(self, inp):
        inp = self.layer1(inp)
        inp = self.bn(inp)
        inp = self.sigmoid(inp)
        return self.layer2(inp)


class MLP_Sigmoid(torch.nn.Module):
    def __init__(self):
        super(MLP_Sigmoid, self).__init__()
        self.layer1 = torch.nn.Linear(784, 800, bias=False)
        self.sigmoid = torch.nn.Sigmoid()
        self.layer2 = torch.nn.Linear(800, 10)

    def forward(self, inp):
        inp = self.layer1(inp)
        inp = self.sigmoid(inp)
        return self.layer2(inp)


class MLP_Relu_Bn(torch.nn.Module):
    def __init__(self):
        super(MLP_Relu_Bn, self).__init__()
        self.layer1 = torch.nn.Linear(784, 800, bias=False)
        self.bn = torch.nn.BatchNorm1d(800)
        self.bn.weight.data.fill_(2.5)
        self.bn.bias.data.fill_(2.5)
        self.relu = torch.nn.ReLU()
        self.layer2 = torch.nn.Linear(800, 10)

    def forward(self, inp):
        inp = self.layer1(inp)
        inp = self.bn(inp)
        inp = self.relu(inp)
        return self.layer2(inp)


class MLP_Relu(torch.nn.Module):
    def __init__(self):
        super(MLP_Relu, self).__init__()
        self.layer1 = torch.nn.Linear(784, 800, bias=False)
        self.relu = torch.nn.ReLU()
        self.layer2 = torch.nn.Linear(800, 10)

    def forward(self, inp):
        inp = self.layer1(inp)
        inp = self.relu(inp)
        return self.layer2(inp)


class Ann_Model_Training:
    def __init__(self):
        self.file_path = "./data/mnist/"
        self.EPOCHS = 15
        self.batch_size_train = 600
        self.batch_size_test = 1000
        self.learning_rate = 0.01
        self.random_seed = 1024
        self.log_interval = 10
        self.eps = 1e-8
        self.train_loader = None
        self.test_loader = None
        self.model = None

    def fetch_dataset(self):
        self.train_loader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST(self.file_path, train=True, download=True,
                                       transform=torchvision.transforms.Compose([
                                           torchvision.transforms.RandomRotation(10),
                                           torchvision.transforms.RandomCrop(28, padding=3),
                                           torchvision.transforms.ToTensor(),
                                           torchvision.transforms.Normalize(
                                               (0.1307,), (0.3081,))
                                       ])),
            batch_size=self.batch_size_train, shuffle=True)

        self.test_loader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST(self.file_path, train=False, download=True,
                                       transform=torchvision.transforms.Compose([
                                           torchvision.transforms.ToTensor(),
                                           torchvision.transforms.Normalize(
                                               (0.1307,), (0.3081,))
                                       ])),
            batch_size=self.batch_size_test, shuffle=True)

    def train_process(self, model, optimizer, criterion):
        if self.train_loader is None:
            self.fetch_dataset()
        for epoch in range(self.EPOCHS):
            for batch_idx, (data, target) in enumerate(self.train_loader):
                optimizer.zero_grad()
                data = data.view(data.size(0), -1)
                out = model(data)
                loss = criterion(out, target) + self.var_norm(model, 0.1)
                loss.backward()
                optimizer.step()
                if batch_idx % self.log_interval == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(self.train_loader.dataset),
                               100 * batch_idx / len(self.train_loader), loss.item()))
        return model

    def test_process(self, net, mode="Test"):
        if self.test_loader is None or self.train_loader is None:
            self.fetch_dataset()
        if mode == "Test":
            data_loader = self.test_loader
        else:
            data_loader = self.train_loader
        net.eval()
        test_loss = 0
        correct = 0

        with torch.no_grad():
            for data, target in data_loader:
                data = data.view(data.size(0), -1)
                out1 = net(data)
                test_loss += torch.nn.functional.cross_entropy(out1, target, reduction="sum").item()
                pred = out1.data.max(1, keepdim=True)[1]
                correct += torch.sum(pred.eq(target.data.view_as(pred)))

        test_loss /= len(data_loader.dataset)
        ans = '\n Model: {:} \n {:} set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            str(type(net)), mode, test_loss, correct, len(data_loader.dataset),
            100. * correct / len(data_loader.dataset))
        print(ans)
        return ans

    def training_model(self, fix_seed=True, save_op=True):
        if fix_seed:
            torch.manual_seed(self.random_seed)
        print("\n------ MNIST MLP_SIGMOID_BN Start------")
        net1 = MLP_Sigmoid_Bn()
        optimizer = torch.optim.Adam(net1.parameters(), lr=self.learning_rate)
        net1.train()
        criterion = torch.nn.CrossEntropyLoss()

        net1 = self.train_process(net1, optimizer=optimizer, criterion=criterion)

        print("\n------ MNIST MLP_SIGMOID Start------")
        net2 = MLP_Sigmoid()
        optimizer = torch.optim.Adam(net2.parameters(), lr=self.learning_rate)
        net2.train()
        criterion = torch.nn.CrossEntropyLoss()
        print("\n------ MNIST MLP_SIGMOID Start------")
        net2 = self.train_process(net2, optimizer=optimizer, criterion=criterion)

        net3 = MLP_Relu_Bn()
        optimizer = torch.optim.Adam(net3.parameters(), lr=self.learning_rate)
        net3.train()
        criterion = torch.nn.CrossEntropyLoss()
        print("\n------ MNIST MLP_Relu_Bn Start------")
        net3 = self.train_process(net3, optimizer=optimizer, criterion=criterion)

        net4 = MLP_Relu()
        optimizer = torch.optim.Adam(net4.parameters(), lr=self.learning_rate)
        net4.train()
        criterion = torch.nn.CrossEntropyLoss()
        print("------ MNIST MLP_Relu_Bn Start------")
        net4 = self.train_process(net4, optimizer=optimizer, criterion=criterion)

        if save_op:
            torch.save(net1.state_dict(), "mlp_sigmoid_bn.pt")
            torch.save(net2.state_dict(), "mlp_sigmoid.pt")
            torch.save(net3.state_dict(), "mlp_relu_bn.pt")
            torch.save(net4.state_dict(), "mlp_relu.pt")

        mode = "Train"
        ans = self.test_process(net1) 
        ans += self.test_process(net1, mode=mode)
        with open("mlp_sigmoid_bn_test.bin", "wb") as f:
            pickle.dump(ans, f)
        ans = self.test_process(net2)
        ans += self.test_process(net2, mode=mode)
        with open("mlp_sigmoid_test.bin", "wb") as f:
            pickle.dump(ans, f)
        ans = self.test_process(net3)
        ans += self.test_process(net3, mode=mode)
        with open("mlp_relu_bn_test.bin", "wb") as f:
            pickle.dump(ans, f)
        ans = self.test_process(net4)
        ans += self.test_process(net4, mode=mode)
        with open("mlp_relu_test.bin", "wb") as f:
            pickle.dump(ans, f)

    def test_models(self, mode="Test"):
        net1 = MLP_Sigmoid_Bn()
        state = torch.load("mlp_sigmoid_bn.pt")
        net1.load_state_dict(state)

        net2 = MLP_Sigmoid()
        state = torch.load("mlp_sigmoid.pt")
        net2.load_state_dict(state)

        net3 = MLP_Relu_Bn()
        state = torch.load("mlp_relu_bn.pt")
        net3.load_state_dict(state)

        net4 = MLP_Relu()
        state = torch.load("mlp_relu.pt")
        net4.load_state_dict(state)

        _ = self.test_process(net1, mode=mode)
        _ = self.test_process(net2, mode=mode)
        _ = self.test_process(net3, mode=mode)
        _ = self.test_process(net4, mode=mode)

    @staticmethod
    def var_norm(model: torch.nn.Module, norm_lambda: float) -> torch.Tensor:
        var_reg = torch.tensor(0.)
        for param in model.parameters():
            var_reg += torch.var(param)
        return norm_lambda * var_reg


if __name__ == "__main__":
    net = MLP_Sigmoid_Bn()
    tool = Ann_Model_Training()
    tool.fetch_dataset()
    optimizer = torch.optim.Adam(net.parameters())
    criterion = torch.nn.CrossEntropyLoss()
    net = tool.train_process(net, optimizer, criterion)
    _ = tool.test_process(net)

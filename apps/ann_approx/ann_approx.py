from mnn_core.mnn_modules import *


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.ln1_mean = torch.nn.Linear(1, 100)
        self.bn1_mean = torch.nn.BatchNorm1d(100)
        self.bn1_std = torch.nn.BatchNorm1d(100)
        self.ln1_std = torch.nn.Linear(1, 100)
        self.ln2 = torch.nn.Linear(100, 50)
        self.bn2 = torch.nn.BatchNorm1d(50)
        self.ln3 = torch.nn.Linear(50, 1)

    def forward(self, x, y):
        x = torch.sigmoid(self.bn1_mean(self.ln1_mean(x)))
        y = torch.sigmoid(self.bn1_std(self.ln1_std(y)))
        z = x+y
        z = torch.sigmoid(self.bn2(self.ln2(z)))
        z = self.ln3(z)
        return z


class Train_Model:
    def __init__(self, epochs=10000, lr=1e-2, schedule=None, base_step=1,
                 up_bound=50, low_bound=-50, log_interval=50):
        if schedule is None:
            schedule = [1, 0.1, 0.01]
        self.EPOCH = epochs
        self.LR = lr
        self.SCHEDULE = schedule
        self.BASE = base_step
        self.UP = up_bound
        self.LOW = low_bound
        self.log_interval = log_interval

    @staticmethod
    def synthetic_data(low=-50, up=50, step=1):
        u = torch.arange(low, up, step, dtype=torch.float64)
        s = torch.arange(0, up - low, step, dtype=torch.float64)
        uu, ss = torch.meshgrid(u, s)
        uu = uu.reshape(-1, 1)
        ss = ss.reshape(-1, 1)
        tu = Mnn_Activate_Mean.apply(uu, ss)
        return uu, ss, tu

    def training(self, save_name="approx.pt"):
        net = Net()
        optimizer = torch.optim.AdamW(net.parameters(), self.LR)
        criterion = torch.nn.MSELoss()
        net.train()
        with open("log_{:}.txt".format(save_name[0:-3]), "a+", encoding="utf-8") as f:
            f.write("\n ==========Train Start=========\n")
        for i in self.SCHEDULE:
            step = self.BASE * i
            u, s, target = self.synthetic_data(self.LOW, self.UP, step)
            for epoch in range(self.EPOCH):
                optimizer.zero_grad()
                out = net(u, s)
                loss = criterion(out, target)
                loss.backward()
                if epoch % self.log_interval == 0:
                    with open("log_{:}.txt".format(save_name[0:-3]), "a+", encoding="utf-8") as f:
                        train_info = "\nSchedule:{:}, Epoch: {:}, Loss:{:}".format(i, epoch, loss.item())
                        f.write(train_info)
            torch.save(net.state_dict(), save_name)


if __name__ == "__main__":
    tool = Train_Model()
    tool.training()




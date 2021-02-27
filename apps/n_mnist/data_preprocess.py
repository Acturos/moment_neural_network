import os
import pickle
from Mnn_Core.mnn_modules import *


class Event():
    '''
    This class provides a way to store, read, write and visualize spike event.

    Members:
        * ``x`` (numpy ``int`` array): `x` index of spike event.
        * ``y`` (numpy ``int`` array): `y` index of spike event (not used if the spatial dimension is 1).
        * ``p`` (numpy ``int`` array): `polarity` or `channel` index of spike event.
        * ``t`` (numpy ``double`` array): `timestamp` of spike event. Time is assumend to be in ms.

    Usage:

    >>> TD = Event(xEvent, yEvent, pEvent, tEvent)
    '''

    def __init__(self, xEvent, yEvent, pEvent, tEvent):
        if yEvent is None:
            self.dim = 1
        else:
            self.dim = 2

        self.x = xEvent if type(xEvent) is np.ndarray else np.asarray(xEvent)  # x spatial dimension
        self.y = yEvent if type(yEvent) is np.ndarray else np.asarray(yEvent)  # y spatial dimension
        self.p = pEvent if type(pEvent) is np.ndarray else np.asarray(pEvent)  # spike polarity
        self.t = tEvent if type(tEvent) is np.ndarray else np.asarray(tEvent)  # time stamp in ms

        if not issubclass(self.x.dtype.type, np.integer): self.x = self.x.astype('int')
        if not issubclass(self.p.dtype.type, np.integer): self.p = self.p.astype('int')

        if self.dim == 2:
            if not issubclass(self.y.dtype.type, np.integer): self.y = self.y.astype('int')

        self.p -= self.p.min()


def read2Dspikes(file_path):
    '''
    Reads two dimensional binary spike file and returns a TD event.
    It is the same format used in neuromorphic datasets NMNIST & NCALTECH101.

    The binary file is encoded as follows:
        * Each spike event is represented by a 40 bit number.
        * First 8 bits (bits 39-32) represent the xID of the neuron.
        * Next 8 bits (bits 31-24) represent the yID of the neuron.
        * Bit 23 represents the sign of spike event: 0=>OFF event, 1=>ON event.
        * The last 23 bits (bits 22-0) represent the spike event timestamp in microseconds.

    Arguments:
        * ``filename`` (``string``): path to the binary file.

    Usage:

    >>> TD = read2Dspikes(file_path)
    '''
    with open(file_path, 'rb') as inputFile:
        inputByteArray = inputFile.read()
    inputAsInt = np.asarray([x for x in inputByteArray])
    xEvent = inputAsInt[0::5]
    yEvent = inputAsInt[1::5]
    pEvent = inputAsInt[2::5] >> 7
    tEvent = ((inputAsInt[2::5] << 16) | (inputAsInt[3::5] << 8) | (inputAsInt[4::5])) & 0x7FFFFF
    return Event(xEvent, yEvent, pEvent, tEvent / 1000)  # convert spike times to ms


def raw2Tensor(TD: Event, frameRate=1000):
    if TD.dim != 2:
        raise Exception('Expected Td dimension to be 2. It was: {}'.format(TD.dim))
    interval = 1e3 / frameRate  # in ms
    xDim = TD.x.max() + 1
    yDim = TD.y.max() + 1
    if xDim != 34 or yDim != 34 or xDim != yDim:
        print(xDim, yDim)
        raise ValueError

    minFrame = int(np.floor(TD.t.min() / interval))
    maxFrame = int(np.ceil(TD.t.max() / interval))
    samples = maxFrame - minFrame
    raw_data = torch.zeros(samples, yDim, xDim)
    for i in range(samples):
        tStart = (i + minFrame) * interval
        tEnd = (i + minFrame + 1) * interval
        timeMask = (TD.t >= tStart) & (TD.t < tEnd)
        positive = (timeMask & (TD.p == 1))
        negative = (timeMask & (TD.p == 0))
        raw_data[i, TD.y[positive], TD.x[positive]] = 1
        raw_data[i, TD.y[negative], TD.x[negative]] = -1

    x = raw_data.view(samples, 1, -1)
    y = torch.transpose(x.clone().detach(), dim0=-2, dim1=-1)

    x_mean = torch.mean(x, dim=0)
    x_std = torch.std(x, dim=0)

    y_mean = torch.mean(y, dim=0)

    vx = x - x_mean
    vy = y - y_mean

    correlation = torch.sum(vx * vy, dim=0) / torch.sqrt(torch.sum(vx ** 2, dim=0) * torch.sum(vy ** 2, dim=0))
    temp = torch.zeros_like(correlation)
    correlation = torch.where(torch.isnan(correlation), temp, correlation)
    correlation = correlation.fill_diagonal_(1.0)
    
    # fire rate in second
    return x_mean.view(-1) * (1e3/interval), x_std.view(-1) * (1e3/interval), correlation


class nmistDataset(torch.utils.data.Dataset):
    def __init__(self, datasetPath="D:/Data_repos/N-MNIST/data/", mode: str = "train", frames=1000):
        super(nmistDataset, self).__init__()
        self.path = datasetPath
        self.mode = mode
        self.file_path = None
        self.labels = None
        self.frames = frames
        self._fetch_files_path()

    def _fetch_files_path(self):
        data_dir = self.path + self.mode + "/"
        files_name = []
        labels = []
        for i in os.listdir(data_dir):
            next_dir = data_dir + i
            for j in os.listdir(next_dir):
                labels.append(i)
                files_name.append(data_dir + i + "/" + j)
        self.file_path = files_name
        self.labels = labels

    def remove_outlier(self):
        totol_sample = len(self.file_path)
        remove = 0
        correct_file = []
        correct_label = []
        for item in range(len(self.file_path)):
            file = self.file_path[item]
            td = read2Dspikes(file)
            xDim = td.x.max() + 1
            yDim = td.y.max() + 1
            if xDim != 34 or yDim != 34 or xDim != yDim:
                print(file)
                remove += 1
                if os.path.exists(file):
                    os.remove(file)
                else:
                    print("The file {:} does not exist".format(file))
            else:
                correct_file.append(file)
                correct_label.append(self.labels[item])
        self.file_path = correct_file
        self.labels = correct_label
        print("For {:} dataset, before check:{:}, remove: {:}".format(self.mode, totol_sample, remove))

    def __getitem__(self, item):
        input_sample = self.file_path[item]
        class_label = eval(self.labels[item])

        u, s, r = raw2Tensor(read2Dspikes(input_sample), frameRate=self.frames)

        return u, s, r, class_label

    def __len__(self):
        return len(self.file_path)
    
    
def data_preprocess(data_path, mode, fps=1000):
    dataset = nmistDataset(datasetPath=data_path, mode=mode, frames=fps)
    save_path = "./data/processed_nmnist/"
    for i in range(len(dataset)):
        u, s, r, t = dataset[i]
        file_name = save_path + mode + "/" + str(t) + "/" + str(t) + "_" + str(i) + ".bin"
        with open(file_name, "wb") as f:

            pickle.dump((u.detach(), s.detach(), r.detach(), t), f)


class Mnn_MLP_with_corr(torch.nn.Module):
    def __init__(self, hidden=800, ln1_std=True, ln2_bias=True, ln2_std=False):
        super(Mnn_MLP_with_corr, self).__init__()
        self.layer1 = Mnn_Linear_Module_with_Rho(34*34, hidden, bn_ext_std=ln1_std)
        self.layer2 = Mnn_Summation_Layer_with_Rho(hidden, 10, bias=ln2_bias, ext_bias_std=ln2_std)

    def forward(self, ubar, sbar, rho):
        ubar, sbar, rho = self.layer1(ubar, sbar, rho)
        ubar, sbar, rho = self.layer2(ubar, sbar, rho)
        return ubar, sbar, rho


class myDataset(torch.utils.data.Dataset):
    def __init__(self, path="./data/processed_nmnist/", mode="train"):
        super(myDataset, self).__init__()
        self.path = path
        self.mode = mode
        self.file_path = None
        self._fetch_file_path()

    def _fetch_file_path(self):
        data_dir = self.path + self.mode + "/"
        files_name = []
        for i in os.listdir(data_dir):
            next_dir = data_dir + i
            for j in os.listdir(next_dir):
                files_name.append(data_dir + i + "/" + j)
        self.file_path = files_name

    def __getitem__(self, item):
        path = self.file_path[item]
        with open(path, "rb") as f:
            u, s, r, t = pickle.load(f)

        return u, s, r, t

    def __len__(self):
        return len(self.file_path)


class Training_Model:
    def __init__(self):
        self.path = "./data/n_mnist/"
        self.EPOCHS = 9
        self.BATCH = 256
        self.lr = 1e-2
        self.log_interval = 20
        self.seed = 1024
        self.train_loader = None
        self.test_loader = None
        self.model = None
        self.net_params = None
        self.fps = 500
        self.loss_mode = 0
        self.fetch_dataset()
        self.eps = 1e-8

    def fetch_dataset(self):
        self.train_loader = torch.utils.data.DataLoader(dataset=nmistDataset(self.path, mode="train", frames=self.fps),
                                                        batch_size=self.BATCH, shuffle=True, num_workers=4)

        self.test_loader = torch.utils.data.DataLoader(dataset=nmistDataset(self.path, mode="test", frames=self.fps),
                                                       batch_size=self.BATCH, shuffle=True, num_workers=4)

    def train_process(self, model, data_loader, criterion, epochs, model_name, optimizer):
        model.train()
        for epoch in range(epochs):
            for batch_idx, (mean, std, rho, target) in enumerate(data_loader):
                optimizer.zero_grad()
                out1, out2, out3 = model(mean, std, rho)
                if self.loss_mode == 0:
                    loss = criterion(out1, target)
                else:
                    loss = criterion(out1 / (out2 + self.eps), target)
                loss.backward()
                optimizer.step()
                if batch_idx % self.log_interval == 0:
                    train_info = '\nTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(mean), len(self.train_loader.dataset),
                               100 * batch_idx / len(self.train_loader), loss.item())
                    with open("log_{:}.txt".format(model_name[0:-3]), "a+", encoding="utf-8") as f:
                        f.write(train_info)
        return model

    def training(self, save_name="nmnist_mlp_1000fps_v1.pt", fix_seed=False):
        if fix_seed:
            torch.manual_seed(seed=self.seed)
        if self.net_params is None:
            raise ValueError
        else:
            hidden, ln1_std, ln2_bias, ln2_std = self.net_params
        net = Mnn_MLP_with_corr(hidden=hidden, ln1_std=ln1_std, ln2_bias=ln2_bias, ln2_std=ln2_std)
        torch.save(net.state_dict(), "init_params_" + save_name)
        criterion = torch.nn.CrossEntropyLoss()
        with open("log_{:}.txt".format(save_name[0:-3]), "a+", encoding="utf-8") as f:
            f.write("------N-MNIST MNN TRAINING START--------\n")
            f.write("Net: {:}, hidden: {}, ln1_std:{}, ln2_bias{}, ln2_std:{}, loss_mode:{:}\n".format(
                save_name[0:-3], hidden, ln1_std, ln2_bias, ln2_std, self.loss_mode))
        best = 0
        optimizer = torch.optim.Adam(net.parameters(), lr=self.lr)
        for epoch in range(self.EPOCHS):
            net = self.train_process(net, self.train_loader, criterion, 1, save_name, optimizer)
            correct = self.test_process(net, self.test_loader, save_name)
            if correct > best:
                best = correct
                torch.save(net.state_dict(), save_name)

    def load_model(self, model_name):
        if self.net_params is None:
            raise ValueError
        else:
            hidden, ln1_std, ln2_bias, ln2_std = self.net_params
        net = Mnn_MLP_with_corr(hidden=hidden, ln1_std=ln1_std, ln2_bias=ln2_bias, ln2_std=ln2_std)
        state = torch.load(model_name)
        net.load_state_dict(state)
        return net

    def test_process(self, net, data_loader, model_name: str) -> int:
        net.eval()
        test_loss = 0
        correct: int = 0
        with torch.no_grad():
            for mean, std, rho, target in data_loader:
                out1, out2, out3 = net(mean, std, rho)
                if self.loss_mode == 0:
                    test_loss += F.cross_entropy(out1, target, reduction="sum").item()
                    pred = out1.data.max(1, keepdim=True)[1]
                else:
                    pred = out1 / (out2 + self.eps)
                    test_loss += F.cross_entropy(out1/pred, target, reduction="sum").item()
                    pred = pred.data.max(1, keepdim=True)[1]
                correct += torch.sum(pred.eq(target.data.view_as(pred)))
        test_loss /= len(data_loader.dataset)
        test_info = '\nTest set, Model: {:},  Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            str(model_name), test_loss, correct, len(data_loader.dataset), 100. * correct / len(data_loader.dataset))
        with open("log_{:}.txt".format(model_name[0:-3]), "a+", encoding="utf-8") as f:
            f.write(test_info)
        return correct

    def testing(self, model_name="nmnist_mlp_1000fps_v1.pt", mode="test"):

        net = self.load_model(model_name)
        if mode == "test":
            data_loader = self.test_loader
        else:
            data_loader = self.train_loader

        _ = self.test_process(net, data_loader=data_loader)

    def test_model_with_fps(self, model_name: str, dataset: nmistDataset):
        net = self.load_model(model_name)
        correct = 0
        net.eval()
        with torch.no_grad():
            for i in range(len(dataset)):
                u, s, r, t = dataset[i]
                u = u.view(1, -1)
                s = s.view(1, -1)
                r = r.view(1, 34*34, -1)
                out1, out2, out3 = net(u, s, r)
                if self.loss_mode == 0:
                    pred = out1.data.max(1, keepdim=True)[1]
                else:
                    pred = (out1 / (out2 + self.eps)).data.max(1, keepdim=True)[1]
                if pred == t:
                    correct += 1
                else:
                    continue
        print('\nModel: {:}, {:} set,  Accuracy: {}/{} ({:.0f}%)\n'.format(
            model_name[0:-3], "Test", correct, len(dataset), 100. * correct / len(dataset)))

    def continue_training(self, model_name="mode_v2.pt"):
        net = self.load_model(model_name)
        optimizer = torch.optim.AdamW(net.parameters(), lr=self.lr)
        criterion = torch.nn.CrossEntropyLoss()
        best = 0
        save_name = "CL_" + model_name
        with open("log_{:}.txt".format(model_name[0:-3]), "a+", encoding="utf-8") as f:
            f.write("------N-MNIST MNN  Continue TRAINING START--------\n")
        for _ in range(self.EPOCHS):
            net = self.train_process(net, self.train_loader, criterion, 1, model_name, optimizer)
            accuracy = self.test_process(net, self.test_loader, model_name)
            if accuracy > best:
                best = accuracy
                torch.save(net.state_dict(), save_name)


if __name__ == "__main__":
    tool = Training_Model()
    tool.EPOCHS = 15
    tool.loss_mode = 0
    tool.fps = 500
    tool.lr = 1e-3
    tool.BATCH = 128
    tool.fetch_dataset()
    tool.net_params = (900, True, True, False)
    tool.path = "./data/n_mnist/"
    tool.continue_training("fps500.pt")

    """
    tool.net_params = (800, True, True, False)
    data = nmistDataset(datasetPath="./data/n_mnist/", mode="test", frames=500)
    #tool.training(save_name="model_v1.pt")
    tool.test_model_with_fps("model_v1.pt", data)

    tool.loss_mode = 0
    tool.net_params = (1000, True, True, False)
    #tool.training(save_name="mode_v2.pt")
    tool.test_model_with_fps("mode_v2.pt", dataset=data)

    tool.net_params = (800, True, True, True)
    tool.loss_mode = 1
    #tool.training(save_name="mode_v3.pt")
    tool.test_model_with_fps("mode_v3.pt", data)
    
    tool.net_params = (1000, True, True, True)
    tool.loss_mode = 1
    #tool.training(save_name="mode_v4.pt")
    tool.test_model_with_fps("mode_v4.pt", data)
    """


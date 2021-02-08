from Mnn_Core.mnn_modules import *
from PIL import Image
import torchvision
import os
import torch.fft as tfft
from typing import Tuple


class Mnn_Conv2d_fft(torch.nn.Module):
    def __init__(self, dim_x: int, dim_y: int, bias: bool = False, bias_std: bool=False) -> None:
        super(Mnn_Conv2d_fft, self).__init__()
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.weight = Parameter(torch.Tensor(dim_x, dim_y))
        if bias:
            self.bias = Parameter(torch.Tensor(dim_x, dim_y))
        else:
            self.register_parameter("bias", None)
        if bias_std:
            self.ext_bias_std = Parameter(torch.Tensor(dim_x*dim_y))
        else:
            self.register_parameter("ext_bias_std", None)
        self.reset_parameters()

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}, ext_bias_std={}'.format(
            self.dim_x, self.dim_y, self.bias is not None, self.ext_bias_std is not None
        )

    def reset_parameters(self) -> None:
            init.kaiming_uniform_(self.weight, a=np.sqrt(5))
            if self.bias is not None:
                fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
                bound = 1 / np.sqrt(fan_in)
                init.uniform_(self.bias, -bound, bound)
            if self.ext_bias_std is not None:
                fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
                bound = 1 / np.sqrt(fan_in)
                init.uniform_(self.ext_bias_std, 0, bound)

    def forward(self, mean_in: Tensor, std_in: Tensor, corr_in: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        assert mean_in.size() == std_in.size() and mean_in.dim() >= 2
        wfft = tfft.fft(tfft.fft(self.weight, dim=0), dim=1)
        # mean shape = (N, N)
        if mean_in.dim() == 2:
            corr_in = corr_in.view(self.dim_x, self.dim_y, self.dim_x, self.dim_y)
            cov_in = std_in.unsqueeze(0).unsqueeze(0) * corr_in * std_in.unsqueeze(-1).unsqueeze(-1)
            mean_in_fft = tfft.fft(tfft.fft(mean_in, dim=0), dim=1)
            mean_out = torch.real(tfft.ifft(tfft.ifft(wfft*mean_in_fft, dim=0), dim=1))
            for d in range(4):
                cov_in = tfft.fft(cov_in, dim=d)
            cov_out = wfft.unsqueeze(0).unsqueeze(0) * cov_in * wfft.unsqueeze(-1).unsqueeze(-1)
            for d in range(4):
                cov_out = tfft.ifft(cov_out, dim=d)
            cov_out = torch.real(cov_out)
            cov_out = cov_out.view(self.dim_x * self.dim_y, self.dim_x * self.dim_y)
        # mean shape = (Batch, N, N)
        else:
            num_batch = mean_in.size()[0]
            corr_in = corr_in.view(num_batch, self.dim_x, self.dim_y, self.dim_x, self.dim_y)
            cov_in = std_in.unsqueeze(1).unsqueeze(1) * corr_in * std_in.unsqueeze(-1).unsqueeze(-1)
            mean_in_fft = tfft.fft(tfft.fft(mean_in, dim=1), dim=2)
            mean_out = torch.real(tfft.ifft(tfft.ifft(wfft*mean_in_fft, dim=1), dim=2))
            for d in range(1, 5):
                cov_in = tfft.fft(cov_in, dim=d)
            cov_out = wfft.unsqueeze(0).unsqueeze(0) * cov_in * wfft.unsqueeze(-1).unsqueeze(-1)
            for d in range(1, 5):
                cov_out = tfft.ifft(cov_out, dim=d)
            cov_out = torch.real(cov_out)
            cov_out = cov_out.view(num_batch, self.dim_x*self.dim_y, self.dim_x*self.dim_y)
        if self.bias is not None:
            mean_out += self.bias
        std_out, corr_out = update_correlation(cov_out, self.ext_bias_std)
        std_out = std_out.view(std_in.size())
        return mean_out, std_out, corr_out


class Net(torch.nn.Module):
    def __init__(self, dim_x: int, dim_y: int, num_layers: int, trunk_idx: int, ext_bias_std=False) -> None:
        super(Net, self).__init__()
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.num_layers = num_layers
        self.truck_idx = trunk_idx
        self.ext_bias_std = ext_bias_std
        self.cov1 = Mnn_Conv2d_fft(dim_x, dim_y)
        self.bn1 = Mnn_BatchNorm1d_with_Rho(dim_x * dim_y, ext_std=ext_bias_std)
        self.cov2 = Mnn_Conv2d_fft(dim_x, dim_y)
        self.bn2 = Mnn_BatchNorm1d_with_Rho(dim_x * dim_y, ext_std=ext_bias_std)
        self.a1 = Mnn_Activate_Mean.apply
        self.a2 = Mnn_Activate_Std.apply
        self.a3 = Mnn_Activate_Corr.apply

    def forward(self, ubar: Tensor, sbar: Tensor, rho: Tensor) -> Tuple[list, list, list]:
        ubar, sbar, rho = self.cov1(ubar, sbar, rho)
        if ubar.dim() == 2:
            shape = ubar.size()
            ubar = ubar.view(-1)
            sbar = sbar.view(-1)
            ubar, sbar, rho = self.bn1(ubar, sbar, rho)
        else:
            batch = ubar.size()[0]
            shape = ubar.size()
            ubar = ubar.view(batch, -1)
            sbar = sbar.view(batch, -1)
            ubar, sbar, rho = self.bn1(ubar, sbar, rho)

        uhat = self.a1(ubar, sbar)
        shat = self.a2(ubar, sbar, uhat)
        rho_hat = self.a3(rho, ubar, sbar, uhat, shat)
        uhat = uhat.view(shape)
        shat = shat.view(shape)

        output_u = list()
        output_s = list()
        output_r = list()
        for i in range(self.num_layers):
            ubar, sbar, rho = self.cov2(uhat, shat, rho_hat)
            if ubar.dim() == 2:
                shape = ubar.size()
                ubar = ubar.view(-1)
                sbar = sbar.view(-1)
                ubar, sbar, rho = self.bn2(ubar, sbar, rho)
            else:
                batch = ubar.size()[0]
                shape = ubar.size()
                ubar = ubar.view(batch, -1)
                sbar = sbar.view(batch, -1)
                ubar, sbar, rho = self.bn2(ubar, sbar, rho)

            uhat = self.a1(ubar, sbar)
            shat = self.a2(ubar, sbar, uhat)
            rho_hat = self.a3(rho, ubar, sbar, uhat, shat)
            uhat = uhat.view(shape)
            shat = shat.view(shape)
            if self.training:
                if i < self.truck_idx:
                    output_u.append(uhat)
                    output_s.append(shat)
                    output_r.append(rho_hat)
                else:
                    uhat = uhat.detach()
                    shat = shat.detach()
                    rho_hat = rho_hat.detach()
            else:
                output_u.append(uhat)
                output_s.append(shat)
                output_r.append(rho_hat)

        return output_u, output_s, output_r


class PrepareData:
    def __init__(self, kernel: int, img_size=(256, 256)):
        self.img_size = img_size
        self.kernel = kernel
        self.noise_strength = 0.1
        self.avg_pool = torch.nn.AvgPool2d(kernel_size=kernel, stride=kernel)
        self.transformer = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()
        ])

    # img size: (384, 286) -> (256, 256)
    def read_img(self, img_path: str):
        im = Image.open(img_path)
        im = im.resize(self.img_size)
        return im

    def crop_img2tensor(self, img_path):
        img = self.read_img(img_path)
        width, height = img.size
        assert width == height
        kernel = self.kernel
        count = int(width/kernel)
        box_list = list()
        for i in range(count):
            for j in range(count):
                # (left, upper, right, lower)
                box = (j*kernel, i*kernel, (j+1)*kernel, (i+1)*kernel)
                box_list.append(box)
        img_list = [self.transformer(img.crop(box)).view(1, -1) for box in box_list]
        img_list = torch.cat(img_list, dim=0)
        return img_list

    @staticmethod
    def compute_correlation(img_patches: Tensor) -> Tensor:
        patches = img_patches.size()[0]
        correlation = torch.zeros(patches, patches)
        for i in range(patches):
            for j in range(patches):
                x = img_patches[i].clone().detach().view(-1, 1)
                #print(x.shape)
                y = img_patches[j].clone().detach().view(1, -1)
                correlation[i, j] = (torch.mean(torch.mm(x, y)) - torch.mean(x)*torch.mean(y)) / (torch.std(x) * torch.std(y))
        correlation = correlation.fill_diagonal_(1.0)
        return correlation

    def data_preprocess(self, img_path: str, add_noise=False, need_correlation=False):
        img_list = self.crop_img2tensor(img_path)
        img_list = img_list.type(torch.DoubleTensor)
        # rescale value into [-1, 1]
        img_list = 2*(img_list - torch.min(img_list)) / (torch.max(img_list)) - 1
        if add_noise:
            img_list += torch.randn(img_list.size()) * self.noise_strength
        mean = torch.mean(img_list, dim=1)
        std = torch.std(img_list, dim=1)
        if need_correlation:
            rho = self.compute_correlation(img_list)
            return mean, std, rho
        else:
            return mean, std

    def fetch_train_data(self, img_path: str, seq_len: int):
        im = self.read_img(img_path)
        raw_tensor = self.transformer(im)
        mean = self.avg_pool(raw_tensor)
        std = torch.sqrt(self.avg_pool(raw_tensor ** 2) - mean ** 2)
        total_neuron = int((self.img_size[0] / self.kernel) ** 2)
        rho = torch.eye(total_neuron)

        mean = mean.repeat(seq_len, 1, 1)
        std = std.repeat(seq_len, 1, 1)
        rho = rho.repeat(seq_len, 1, 1)

        train_mean = mean + torch.randn(mean.size()) * torch.mean(mean) * self.noise_strength
        train_std = torch.abs(std + torch.randn(std.size()) * torch.mean(std) * self.noise_strength)
        train_rho = rho.clone()

        return mean, std, rho, train_mean, train_std, train_rho


class Model_Training:
    def __init__(self, base_lr=1e-3, max_lr=1e-1, epoch=1000, img_size=256, kernel=16, num_layers=50, truck=10,
                 seq_len=5, train_batch=10):
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.EPOCH = epoch
        self.img_size = img_size
        self.kernel = kernel
        self.num_layers = num_layers
        self.truck = truck
        self.seq_len = seq_len
        self.train_batch = train_batch
        self.tool = PrepareData(self.kernel, img_size=(self.img_size, self.img_size))
        self.dim_x = int(self.img_size / self.kernel)
        self.dim_y = int(self.img_size / self.kernel)
        self.img_dir = "./data/BioID/"
        self.img_path = None
        self.desire_input_dim()
        self.fetch_file_path()

    def fetch_file_path(self):
        file_pah = list()
        for i in os.listdir(self.img_dir):
            file_pah.append(self.img_dir + i)
        self.img_path = file_pah

    def prepare_data(self):
        self.tool = PrepareData(self.kernel, img_size=(self.img_size, self.img_size))

    def desire_input_dim(self):
        self.dim_x = int(self.img_size / self.kernel)
        self.dim_y = int(self.img_size / self.kernel)

    def training(self):
        net = Net(self.dim_x, self.dim_y, num_layers=self.num_layers, trunk_idx=self.truck, ext_bias_std=True)
        optimizer = torch.optim.SGD(net.parameters(), lr=self.max_lr, momentum=0.9)
        schedule = torch.optim.lr_scheduler.CyclicLR(optimizer, self.base_lr, self.max_lr, step_size_up=500)
        criterion = torch.nn.MSELoss()
        net.train()
        loss = torch.tensor(0.)
        for epoch in range(self.EPOCH):
            for path in self.img_path:
                u, s, r, tu, ts, tr = self.tool.fetch_train_data(path, self.seq_len)
                for i in range(self.train_batch):
                    optimizer.zero_grad()
                    loss = torch.tensor(0.)
                    train_u, train_s, train_r = net(u, s, r)
                    for item in range(len(train_u)):
                        loss = loss + criterion(train_u[item], tu) + criterion(train_s[item], ts)\
                               + criterion(train_r[item], tr)
                    loss.backward()
                    optimizer.step()
                    schedule.step()
            if epoch % 10 == 0:
                train_info = "\nTrain Epoch :{:}, Loss: {:}".format(epoch, loss.item())
                with open("hopfiled_log.txt", "a+", encoding="utf-8") as f:
                    print(train_info)
                    f.write(train_info)
        torch.save(net.state_dict(), "hopfield_demo.pt")








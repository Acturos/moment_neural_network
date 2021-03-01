from Mnn_Core.mnn_modules import *
from PIL import Image
import torchvision
import os
import torch.fft as tfft
from typing import Tuple


class Mnn_Conv2d_fft(torch.nn.Module):
    def __init__(self, dim_x: int, dim_y: int, bias: bool = False, bias_std: bool = False) -> None:
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
            ubar = ubar.view(1, -1)
            sbar = sbar.view(1, -1)
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
        for layer_idx in range(self.num_layers):
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
                if layer_idx < self.truck_idx:
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


class myDataset(torch.utils.data.Dataset):
    def __init__(self, kernel: int, img_size: int = 256, path: str = "./data/BioID/") -> None:
        super(myDataset, self).__init__()
        self.path = path
        self.img_size = img_size
        self.kernel = kernel
        self.dim_y = None
        self.dim_x = None
        self.files = None
        self._fetch_file_path()
        self.avg_pool = torch.nn.AvgPool2d(kernel_size=kernel, stride=kernel)
        self.transformer = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.5, ], [0.5, ]),
            torchvision.transforms.RandomErasing(p=0.7, value="random")
        ])
        self.target_transformer = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.5, ], [0.5, ])
        ])
        self.desire_input_dim()

    def _fetch_file_path(self):
        files_name = []
        for i in os.listdir(self.path):
            files_name.append(self.path + i)
        self.files = files_name

    def read_img(self, img_path: str):
        im = Image.open(img_path)
        im = im.resize((self.img_size, self.img_size))
        return im

    def desire_input_dim(self):
        self.dim_x = int(self.img_size / self.kernel)
        self.dim_y = int(self.img_size / self.kernel)

    def __getitem__(self, item):
        im = self.read_img(self.files[item])
        train_im = self.transformer(im)
        train_im = train_im.type(torch.float64)
        target_im = self.target_transformer(im)
        target_im = target_im.type(torch.float64)
        tar_u = self.avg_pool(target_im)
        tar_s = torch.sqrt(torch.abs(self.avg_pool(target_im ** 2) - tar_u ** 2))

        mean = self.avg_pool(train_im)
        std = torch.sqrt(torch.abs(self.avg_pool(train_im ** 2) - mean ** 2))

        tar_u = tar_u.view(self.dim_x, self.dim_y)
        tar_s = tar_s.view(self.dim_x, self.dim_y)
        mean = mean.view(self.dim_x, self.dim_y)
        std = std.view(self.dim_x, self.dim_y)
        rho = torch.eye(self.dim_x * self.dim_y)
        tar_rho = rho.clone().detach()
        return mean, std, rho, tar_u, tar_s, tar_rho

    def __len__(self):
        return len(self.files)


class Model_Training:
    def __init__(self, base_lr=1e-3, epoch=1000, batch=3, img_size=256, kernel=16,
                 num_layers=50, truck=10, train_batch=10, log_interval=50):
        self.lr = base_lr
        self.EPOCH = epoch
        self.BATCH = batch
        self.img_size = img_size
        self.kernel = kernel
        self.num_layers = num_layers
        self.truck = truck
        self.log_interval = log_interval
        self.train_batch = train_batch
        self.img_dir = "./data/BioID/"
        self.train_loader = None
        self.dim_x = None
        self.dim_y = None
        self.desire_input_dim()
        self.fetch_dataset()

    def fetch_dataset(self):
        self.train_loader = torch.utils.data.DataLoader(
            dataset=myDataset(self.kernel, self.img_size, self.img_dir),
            batch_size=self.BATCH, shuffle=True
        )

    def desire_input_dim(self):
        self.dim_x = int(self.img_size / self.kernel)
        self.dim_y = int(self.img_size / self.kernel)

    def train_process(self, model, data_loader, criterion, epochs, model_name, optimizer):
        model.train()
        loss = torch.tensor(0.)
        for epoch in range(epochs):
            for batch_idx, (u, s, r, tu, ts, tr) in enumerate(data_loader):
                optimizer.zero_grad()
                out1, out2, out3 = model(u, s, r)

                loss = torch.tensor(0.)
                for item in range(len(out1)):
                    loss = loss + criterion(out1[item], tu) + criterion(out2[item], ts)\
                               + criterion(out3[item], tr)
                loss.backward()

                optimizer.step()
            if epoch % self.log_interval == 0:
                train_info = "\n Train Epoch {:}, Loss:{:}\n".format(epoch, loss.item())
                with open("log_{:}.txt".format(model_name[0:-3]), "a+", encoding="utf-8") as f:
                    f.write(train_info)
        return model

    def training(self, model_name="hopfield.pt"):
        net = Net(self.dim_x, self.dim_y, num_layers=self.num_layers, trunk_idx=self.truck, ext_bias_std=True)
        optimizer = torch.optim.AdamW(net.parameters(), lr=self.lr)
        criterion = torch.nn.MSELoss()
        with open("log_{:}.txt".format(model_name[0:-3]), "a+", encoding="utf-8") as f:
            f.write("========Train Start=========\n")
        for i in range(self.train_batch):
            net = self.train_process(net, self.train_loader, criterion, self.EPOCH, model_name, optimizer)
            torch.save(net.state_dict(), model_name)


if __name__ == "__main__":
    tool = Model_Training()
    tool.training()




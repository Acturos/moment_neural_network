#from mnn_core.mnn_pytorch import *
from mnn_core.mnn_pytorch import *
import numpy as np
import torch
import time
from torch.utils.tensorboard import SummaryWriter
#from ray import tune
import json
from apps.regression.data import *
from apps.regression.visualization_tools import *

torch.set_default_tensor_type(torch.DoubleTensor)
#seed = 5
#torch.manual_seed(seed)

def loss_mse_covariance(pred_mean, pred_std, pred_corr, target_mean, target_std, target_corr, cov_only = False):    
    pred_cov = pred_std.unsqueeze(1)*pred_corr*pred_std.unsqueeze(2)
    target_cov = target_std.unsqueeze(1)*target_corr*target_std.unsqueeze(2)
    loss2 = F.l1_loss(pred_cov, target_cov)
    #loss2 = F.mse_loss(pred_std, target_std)
    if cov_only: #do not constraint mean activity
        loss = loss2
    else:        
        loss = loss2 + F.mse_loss(pred_mean, target_mean)
    return loss


class MomentLayer(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(MomentLayer, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.linear = Mnn_Linear_Corr(input_size, output_size)

        self.bn_mean = torch.nn.BatchNorm1d(output_size)
        self.bn_mean.weight.data.fill_(2.5)
        self.bn_mean.bias.data.fill_(2.5)
        #this roughly set the input mu in the range (0,5)
        
        self.bn_std = Mnn_Std_Bn1d(output_size , ext_bias = False)
        #self.bn_std.ext_bias.data.fill_(1.0)
        
        #cache the output
        self.mean = 0
        self.std = 0
        self.corr = 0

        return
        
    def forward(self, u, s, rho):
        u, s, rho = self.linear.forward(u, s, rho)
        
        #s = mnn_std_bn1d(self.bn_mean, u, s)
        s = self.bn_std.forward(self.bn_mean, u, s)
        u = self.bn_mean(u)
                
        u_activated = Mnn_Activate_Mean.apply(u, s)
        s_activated = Mnn_Activate_Std.apply(u, s, u_activated)        
        corr_activated = Mnn_Activate_Corr.apply( rho , u, s, u_activated, s_activated)
        
        self.mean, self.std, self.corr = u_activated, s_activated, corr_activated
        
        return u_activated, s_activated, corr_activated

class MomentLayer_no_corr(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(MomentLayer_no_corr, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.linear = Mnn_Linear_without_Corr(input_size, output_size)

        self.bn_mean = torch.nn.BatchNorm1d(output_size)
        self.bn_mean.weight.data.fill_(2.5)
        self.bn_mean.bias.data.fill_(2.5)
        #this roughly set the input mu in the range (0,5)
        
        self.bn_std = Mnn_Std_Bn1d(output_size , ext_bias=False)
        #self.bn_std.ext_bias.data.fill_(1.0)
        
        #cache the output
        self.mean = 0
        self.std = 0
        self.corr = 0

        return
        
    def forward(self, u, s):
        u, s = self.linear.forward(u, s)
        
        #s = mnn_std_bn1d(self.bn_mean, u, s)
        s = self.bn_std.forward(self.bn_mean, u, s)
        u = self.bn_mean(u)
                
        u_activated = Mnn_Activate_Mean.apply(u, s)
        s_activated = Mnn_Activate_Std.apply(u, s, u_activated)        
        
        self.mean, self.std = u_activated, s_activated
        
        return u_activated, s_activated


class MoNet(torch.nn.Module):
    def __init__(self, num_hidden_layers = 10, hidden_layer_size = 64, input_size = 2, output_size = 1):
        super(MoNet, self).__init__()
        self.layer_sizes = [input_size]+[hidden_layer_size]*num_hidden_layers+[output_size]  # input, hidden, output
        
        
        self.layers = torch.nn.ModuleList(
            [MomentLayer(self.layer_sizes[i], self.layer_sizes[i + 1]) for i in range(len(self.layer_sizes) - 1)])
        
        return

    def forward(self, u, s, rho):
        for i in range(len(self.layer_sizes) - 1):
            u, s, rho = self.layers[i].forward(u, s, rho)
        return u, s, rho

class MoNet_no_corr(torch.nn.Module):
    def __init__(self, num_hidden_layers = 10, hidden_layer_size = 64, input_size = 2, output_size = 1):
        super(MoNet_no_corr, self).__init__()
        self.layer_sizes = [input_size]+[hidden_layer_size]*num_hidden_layers+[output_size]  # input, hidden, output
        
        self.layers = torch.nn.ModuleList(
            [MomentLayer_no_corr(self.layer_sizes[i], self.layer_sizes[i + 1]) for i in range(len(self.layer_sizes) - 1)])
        return

    def forward(self, u, s):
        for i in range(len(self.layer_sizes) - 1):
            u, s = self.layers[i].forward(u, s)
        return u, s


class MultilayerPerceptron():
    @staticmethod
    def train(config):        
        if config['seed'] is None:            
            torch.manual_seed(int(time.time())) #use current time as seed
        else:
            torch.manual_seed(config['seed'])
            
        if config['tensorboard']:
            writer = SummaryWriter(log_dir = config['log_dir'] + '_'+ str(config['trial_id']))#log_dir='D:\\mnn_py\\moment_activation\\runs2'
        
        sample_size = config['sample_size']        
        batch_size = config['batch_size'] #64
        num_batches = int(sample_size/batch_size)
        num_epoch = config['num_epoch'] #50#1000
        lr = config['lr']#0.01
        momentum = config['momentum'] #0.9
        optimizer_name = config['optimizer_name']
        input_size = config['input_size']
        output_size = config['output_size']
        
        if config['with_corr']:
            model = MoNet(num_hidden_layers = config['num_hidden_layers'], hidden_layer_size = config['hidden_layer_size'], input_size = input_size, output_size = output_size)
        else:
            model = MoNet_no_corr(num_hidden_layers = config['num_hidden_layers'], hidden_layer_size = config['hidden_layer_size'], input_size = input_size, output_size = output_size)        
            
        train_dataset = Dataset(config['dataset_name'], sample_size = sample_size, input_dim = input_size, output_dim = output_size, with_corr = config['with_corr'], fixed_rho = config['fixed_rho'])
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size)        
    
        model.target_transform = train_dataset.transform
        
        validation_dataset = Dataset(config['dataset_name'], sample_size = 32, input_dim = input_size, output_dim = output_size, transform = train_dataset.transform, with_corr = config['with_corr'], fixed_rho = config['fixed_rho'] )          
        validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size = 32)
            
        
        #model.target_affine = target_affine
        
        #param_container.set_ratio(0.1) # E/I balance. Mostly affect initialization. Mild value seems to improve result.
        #param_container.set_eps(0.01) # 1. Ensures sigma is positive; 2. avoid divide by 0 error. 3. Avoid overflow. Shouldn't need this to be too big. If error still occurs look for other sources.            
        #param_container.print_params()
        
        params = model.parameters()
        
        if optimizer_name == 'Adam':
            optimizer = torch.optim.Adam(params, lr = lr, amsgrad = True) #recommended lr: 0.1 (Adam requires a much smaller learning rate than SGD otherwise won't converge)
        elif optimizer_name == 'SGD':
            optimizer = torch.optim.SGD(params, lr= lr, momentum= momentum) #recommended lr: 2
        else:
            optimizer = torch.optim.Adam(params, lr = lr)
            
        model.checkpoint = {
            'epoch': [],
            'model_state_dict': {},
            'optimizer_state_dict': {},
            'loss': [],
            'dataset_transform_coefs': train_dataset.transform_coefs
            }
        
        t0 = time.perf_counter()
        for epoch in range(num_epoch):            
            model.train()
            #--- iterate ver minibatches
            for i_batch, sample in enumerate(train_dataloader):
                optimizer.zero_grad()                
                if config['with_corr']:
                    u, s, rho = model.forward(sample['input_data'][0], sample['input_data'][1], sample['input_data'][2])
                else:
                    u, s = model.forward(sample['input_data'][0], sample['input_data'][1])
                
                if config['loss'] == 'mse_no_corr':
                    loss = loss_function_mse(u, s, sample['target_data'][0], sample['target_data'][1])
                elif config['loss'] == 'mse_covariance':
                    loss = loss_mse_covariance(u, s, rho, sample['target_data'][0], sample['target_data'][1], sample['target_data'][2])
                loss.backward()
                optimizer.step()
            #--- 
                
            if config['tensorboard']:
                writer.add_scalar("Loss/Train", loss, epoch)
                writer.flush()
            
            #tune.report(loss = loss.item())
            
            if epoch % 1 == 0:
                print('Training epoch {}/{}'.format(epoch,num_epoch))
                with torch.no_grad():
                    model.eval()
                    for i_batch, sample in enumerate(validation_dataloader):
                        if config['with_corr']:
                            u, s, rho = model.forward(sample['input_data'][0], sample['input_data'][1], sample['input_data'][2])
                        else:
                            u, s = model.forward(sample['input_data'][0], sample['input_data'][1])
                        
                        if config['loss'] == 'mse_no_corr':
                            loss = loss_function_mse(u, s, sample['target_data'][0], sample['target_data'][1])
                        elif config['loss'] == 'mse_covariance':
                            loss = loss_mse_covariance(u, s, rho, sample['target_data'][0], sample['target_data'][1], sample['target_data'][2])
                        
                    if config['tensorboard']:
                        writer.add_scalar("Loss/Validation", loss, epoch)
                        writer.flush()
                
                    model.checkpoint['loss'].append(loss.item())
                    model.checkpoint['epoch'].append(epoch)
            
            
        
        #print("===============================")
        print("Number of batches: ", num_batches)
        print("Batch size: ",batch_size)
        print("Learning rate: ", lr)
        print("Momentum: ", momentum)
        print("Time Elapsed: ", time.perf_counter()-t0)
        print("===============================")
        
        model.checkpoint['model_state_dict'] =  model.state_dict()
        model.checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        #writer.add_graph(model, (input_mean,input_std))
        if config['tensorboard']:
            if config['dataset_name'] == 'cue_combo':
                fig, ax1 = plt.subplots(nrows=1, ncols=1)
                l1, = ax1.plot(u.detach().numpy(),s.detach().numpy(),'+')
                l2, = ax1.plot(sample['target_data'][0].detach().numpy(), sample['target_data'][1].detach().numpy(),'.')
                ax1.set_xlabel('Mean')
                ax1.set_ylabel('Std')
                plt.legend([l1,l2],['Output','Target'])    
                
                writer.add_figure('Final output vs target',fig)            
            
                fig1 = VisualizationTools.plot_grid_map(model, with_corr = config['with_corr'], rho = config['fixed_rho'])
                writer.add_figure('Test data grid map',fig1)
            else:
                pass
            
            writer.flush()
            writer.close()
        
        return model

if __name__ == "__main__":    

    config = {'sample_size': 64,
              'batch_size': 32,
              'num_epoch': 10,
              'lr': 0.01,
              'momentum': 0.9,
              'optimizer_name': 'Adam',
              'num_hidden_layers': 3,
              'input_size': 2,
              'output_size': 1,
              'hidden_layer_size': 32,
              'trial_id': int(time.time()),
              'tensorboard': True,
              'log_dir': 'runs/inference',
              'with_corr': True,
              'dataset_name': 'cue_combo',
              'loss':'mse_no_corr',
              'seed': None,
              'fixed_rho': None #ignored if with_corr = False
        }
    
    model = MultilayerPerceptron.train(config)
    
    file_name = config['trial_id']
    torch.save(model.state_dict(), './data/regression/{}.pt'.format(file_name) ) #save result by time stamp
    with open('./data/regression/{}_config.json'.format(file_name),'w') as f:
        json.dump(config,f)
    
    #runfile('./apps/regression/multilayer_perceptron.py', wdir='./')
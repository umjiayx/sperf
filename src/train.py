# train.py
# code for training neural networks
from utils import *
from optimizer import loss_fun, optimizer
from dataloader import sperfloader
from model import NeRF, Siren, nerf2siren
import torch
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn

import os, time, datetime
from tqdm import tqdm
from test import sperftester
import torchvision.utils
import shutil


class sperftrainer():
    def __init__(self):
        # Isaac: super(sperftrainer, self).__init__() # Python 2 style update to Python 3
        # Isaac: In fact, this line does not do too much... 
        super().__init__() 
        parser = config_parser() # Isaac: In utils.py, 
        self.args = parser.parse_args()
        print('config args:', self.args)
        
        # to reproduce the results
        init_env(seed_value=42)
        
        # define a new exp save folder
        if not self.args.expname:
            self.args.expname  = str(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
        self.expdir = os.path.join(self.args.logsdir, self.args.expname)

        loader = sperfloader(projdir=self.args.datadir,
                             psfdir = self.args.psfdir,
                             encoder_level = self.args.encoderlevel,
                             downfactor=self.args.downfactor,
                             start = self.args.start,
                             mode = 'train')
        self.nx, self.nz, self.trainview, self.testview = loader.getsize()
        num_valid = int(len(loader) * self.args.validfrac)
        train, val = data.random_split(loader, [len(loader) - num_valid, num_valid],
                                       generator=torch.Generator().manual_seed(42)) # 
        self.trainloader = data.DataLoader(train,
                                          batch_size=self.args.batchsize,
                                          shuffle=True,
                                          num_workers=8,
                                          pin_memory=True)
        self.validloader = data.DataLoader(val,
                                           batch_size=self.args.batchsize,
                                           shuffle=True,
                                           num_workers=8,
                                           pin_memory=True)
        in_channels = int(6*self.args.encoderlevel)
        if self.args.encoderlevel == 0:
            in_channels = len(loader[0][list(loader[0].keys())[0]]) # 5?
            # in_channels = next(iter(loader[0].values())).shape[0]
        if self.args.model == 'nerf':
            self.model = NeRF(D=self.args.netdepth, # 12
                              W=self.args.netwidth, # 256
                              input_ch=in_channels, # 5
                              output_ch=1,
                              input_ch_views=0,
                              use_viewdirs=False)
        elif self.args.model == 'siren':
            self.model = Siren(hidden_layers=self.args.netdepth,
                              hidden_features=self.args.netwidth,
                              in_features=in_channels,
                              out_features = 1,
                              outermost_linear = True,
                              first_omega_0= self.args.firstomega,
                              hidden_omega_0=self.args.hiddenomega)
        elif self.args.model == 'nerf2siren':
            self.model = nerf2siren(D=self.args.netdepth,
                                    W=self.args.netwidth,
                                    input_ch=in_channels,
                                    output_ch=1,
                                    input_ch_views=0,
                                    use_viewdirs=False)
        else:
            raise NotImplementedError

        print('# of network params: ', count_parameters(self.model))

        ##################################################
        # init the gpu usages
        ##################################################
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = self.args.gpu_ids
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        ##################################################
        # init the training settings
        ##################################################
        self.model.to(self.device)
        self.optimizer = optimizer(self.model, self.args.lr, self.args.optim)
        self.lr_scheduler = ReduceLROnPlateau(self.optimizer, verbose=True, factor=0.2)
        self.writer = SummaryWriter(self.expdir)
        self.criterion = loss_fun(self.args.loss_fun)
        
        ##################################################
        # record code data
        ##################################################
        copytree_code( os.path.join(self.args.srcdir), self.expdir + '/')
        config_path = self.args.config
        shutil.copyfile(config_path, os.path.join(self.expdir, "params.txt"))
        
    def train(self):
        test_init_done, train_init_done = False, False
        
        # write config to the tensorboard 
        self.writer.add_text('init/config', str(self.args))
        
        valid_loss_history = []
        Initial_loss = self.valid()
        print('Initial valid loss: ', Initial_loss)
        for epoch in range(self.args.nepochs):
            print('')
            self.model.train()
            start_t = time.time()
            train_loss = 0
            niter = 0
            for batch in tqdm(self.trainloader):
                self.optimizer.zero_grad()
                for key in batch:
                    # print('key: {}, maximum value: {}'.format(key, torch.max(batch[key]).item()))
                    batch[key] = batch[key].to(self.device)
                pred = self.model.forward(batch['train_grid'])
                loss = self.criterion(pred, batch['train_proj'])
                train_loss += loss.item()
                loss.backward()
                self.optimizer.step()
                niter += 1
            end_t = time.time()
            train_loss = train_loss / niter
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            valid_loss = self.valid()
            valid_loss_history.append(valid_loss)
            self.writer.add_scalar('Loss/valid', valid_loss, epoch)
            print('(Epoch {} / {}) train loss: {:.4f} valid loss: {:.4f} time per epoch: {:.1f}s current lr: {}'.format(
                epoch + 1,
                self.args.nepochs,
                train_loss,
                valid_loss,
                end_t - start_t,
                self.optimizer.param_groups[0]['lr']))
            if (epoch + 1) % 5 == 0:
                print('Save the current model to checkpoint!')
                save_checkpoint(self.model.state_dict(), is_best=False,
                                checkpoint_dir=f'{self.expdir}/models')
            if epoch == np.argmin(valid_loss_history):
                print('The current model is the best model! Save it!')
                save_checkpoint(self.model.state_dict(), is_best=True,
                                checkpoint_dir=f'{self.expdir}/models')
            self.lr_scheduler.step(valid_loss)

            # do online test for visiualization
            if self.args.do_online_test and epoch % self.args.online_test_epoch_gap == 0:
                test_tester = sperftester(f'{self.expdir}/models', 
                                          is_best=False, 
                                          verbose=False, 
                                          args=self.args,
                                          dataset_key='test')
                train_tester = sperftester(f'{self.expdir}/models', 
                                          is_best=False, 
                                          verbose=False, 
                                          args=self.args,
                                          dataset_key='train')
                # print(len(self.tester.testloader))
                self.check_rest(test_tester, epoch, test_init_done, dataset_key='test')
                self.check_rest(train_tester, epoch, train_init_done, dataset_key='train')
                test_init_done = True
                train_init_done = True
        self.writer.close()
    
    def check_rest(self, tester, epoch, init_done, dataset_key='test'):
        pre, gdt = tester.test(dataset_key=dataset_key, denormalize=False)
        
        gdt = torch.from_numpy(gdt)
        
        if dataset_key == 'test':
            gdt = gdt.reshape(self.nx, self.nz, self.testview, 1)
        else:
            gdt = gdt.reshape(self.nx, self.nz, self.trainview, 1)
            
        gdt = torch.permute(gdt, (2, 3, 1, 0)) # BCHW
         # visiualize groud-truth (just do once)
        if not init_done:
            gdt_grid = torchvision.utils.make_grid(gdt, nrow=8, normalize=True, scale_each=True, pad_value=1)
            self.writer.add_image(f'init/{dataset_key}/img_gdt', gdt_grid, epoch, dataformats='CHW')
            self.writer.add_histogram(f'init/{dataset_key}/hist_gdt', gdt, epoch)
        
        # visiualize prediction
        pre = torch.from_numpy(pre)
        if dataset_key == "test":
            pre = pre.reshape(self.nx, self.nz, self.testview, 1)
        else:
            pre = pre.reshape(self.nx, self.nz, self.trainview, 1)
        
        pre = torch.permute(pre, (2, 3, 1, 0)) # BCHW
        pre_grid = torchvision.utils.make_grid(pre, nrow=8, normalize=True, scale_each=True, pad_value=1)
        self.writer.add_image(f'sample/{dataset_key}/img_pre', pre_grid, epoch, dataformats='CHW')
        self.writer.add_histogram(f'sample/{dataset_key}/hist_pre', pre, epoch)
        
        # visiualize prediction snr
        test_snr_list= []
        for i in range(gdt.shape[0]):
            test_snr_list.append(computeSNR(gdt[i, :], pre[i, :]))
        test_snr_avg = torch.mean(torch.stack(test_snr_list))
        self.writer.add_scalar(f'sample/{dataset_key}/snr_pre', test_snr_avg, epoch)
        
        # visiualize difference/error
        dif = torch.abs(pre-gdt)
        dif_grid = torchvision.utils.make_grid(dif, nrow=8, normalize=True, scale_each=True, pad_value=1)
        self.writer.add_image(f'sample/{dataset_key}/img_dif', dif_grid, epoch, dataformats='CHW')
        self.writer.add_histogram(f'sample/{dataset_key}/hist_dif', dif, epoch)
        
        # save to folder
        check_and_mkdir(f'{self.expdir}/{dataset_key}')
        torchvision.utils.save_image(pre_grid, f'{self.expdir}/{dataset_key}/epoch_{epoch:03d}.png')


    def valid(self):
        self.model.eval()
        with torch.no_grad():
            valid_loss = 0
            niter = 0
            for batch in self.validloader:
                for key in batch:
                    batch[key] = batch[key].to(self.device)
                pred = self.model.forward(batch['train_grid'])
                loss = self.criterion(pred, batch['train_proj'])
                valid_loss += loss.item()
            niter += 1
        return valid_loss / niter


if __name__ == "__main__":
    trainer = sperftrainer()
    trainer.train()
    


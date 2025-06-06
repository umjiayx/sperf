# test.py
# code for testing neural networks
from utils import *
from dataloader import sperfloader
from model import NeRF, Siren, nerf2siren
import torch
import torch.utils.data as data
import matplotlib.pyplot as plt
from scipy.io import savemat


class sperftester():
    def __init__(self, checkpoint_dir, 
                is_best = True, 
                verbose=False, 
                args='',
                dataset_key = 'test'):
        super(sperftester, self).__init__()
        if args:
            self.args = args
        else:
            parser = config_parser()
            self.args = parser.parse_args()
        print('config args:', self.args)

        self.loader = sperfloader(projdir=self.args.datadir,
                                  psfdir = self.args.psfdir,
                                  encoder_level=self.args.encoderlevel,
                                  downfactor=self.args.downfactor,
                                  start = self.args.start,
                                  mode = dataset_key)
        self.nx, self.nz, self.trainview, self.testview = self.loader.getsize()

        self.testloader = data.DataLoader(self.loader,
                                           batch_size=self.args.batchsize,
                                           shuffle=False,
                                           num_workers=8,
                                           pin_memory=True)
        in_channels = int(6 * self.args.encoderlevel)
        if self.args.encoderlevel == 0:
            in_channels = len(self.loader[0][list(self.loader[0].keys())[0]])
        if self.args.model == 'nerf':
            self.model = NeRF(D=self.args.netdepth,
                              W=self.args.netwidth,
                              input_ch=in_channels,
                              output_ch=1,
                              input_ch_views=0,
                              use_viewdirs=False)
        elif self.args.model == 'siren':
            self.model = Siren(hidden_layers=self.args.netdepth,
                               hidden_features=self.args.netwidth,
                               in_features=in_channels,
                               out_features=1,
                               outermost_linear=True,
                               first_omega_0=self.args.firstomega,
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
        if verbose:
            print('# of network params: ', count_parameters(self.model))

        if checkpoint_dir:
            print(f'Loading & Testing model from expdir={checkpoint_dir}')
        else:
            print('PLEASE ASSIGN A PATH TO THE MODEL !!!')
            exit(0)
            
        if is_best:
            checkpoint_path = f'{checkpoint_dir}/best_checkpoint.pytorch'
            print('Now loading the best checkpoint!')
        else:
            checkpoint_path = f'{checkpoint_dir}/last_checkpoint.pytorch'
            print('Now loading the last checkpoint!')
        try:
            self.model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
            print('load pre-trained model successfully from: {}!'.format(checkpoint_path))
        except:
            raise IOError(f"load Checkpoint '{checkpoint_path}' failed! ")

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def test(self, dataset_key='test', denormalize = True):
        with torch.no_grad():
            results = []
            GT = []
            self.model.eval()
            for batch in self.testloader:
                for key in batch:
                    batch[key] = batch[key].to(self.device)
                pred = self.model.forward(batch[f'{dataset_key}_grid'])
                results.append(pred.cpu())
                GT.append(batch[f'{dataset_key}_proj'].cpu())
            results_stacked = torch.stack(results[0:-1]).reshape(-1)
            GT_stacked = torch.stack(GT[0:-1]).reshape(-1)
            
            # denomalization
            if denormalize:
                print(f'min val: {self.loader.min_val}, max val: {self.loader.max_val}')
                results_stacked = results_stacked * self.loader.max_val +  self.loader.min_val
                GT_stacked = GT_stacked * self.loader.max_val +  self.loader.min_val
            
            return torch.cat((results_stacked, results[-1].reshape(-1)), dim=0).numpy(), \
                   torch.cat((GT_stacked, GT[-1].reshape(-1)), dim=0).numpy()


if __name__ == "__main__":

    # checkpoint_dir = '/n/higgins/z/xjxu/projects/Sperf/logs/2022-09-21-18-24-46'
    checkpoint_dir = "../logs/fj_c1_s4_b2"
    tester = sperftester(f'{checkpoint_dir}/models')
    results, GT = tester.test(dataset_key = 'test', denormalize = True)
    train_proj = tester.loader.reduced_proj_train
    nx, nz, nview = tester.nx, tester.nz, tester.testview
    print('nview: ', nview)
    results = results.reshape(nx, nz, nview)
    GT = GT.reshape(nx, nz, nview)
    
    test_savedir = f'{checkpoint_dir}/test'
    check_and_mkdir(test_savedir)
    # np.save(f'{test_savedir}/data.npy', 
    #         {'pre':results,'gdt': GT})
    np.save(f'{test_savedir}/pred_proj.npy', results)
    np.save(f'{test_savedir}/gt_proj.npy', GT)
    np.save(f'{test_savedir}/train_proj.npy', train_proj)
    
    # plt.figure()
    # plt.imshow(results[:,:,5].T)
    # plt.show()
    # plt.figure()
    # plt.imshow(GT[:, :, 5].T)
    # plt.show()
    
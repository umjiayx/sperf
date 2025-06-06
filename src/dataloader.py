# dataloader.py
# load and pre-processing data
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.utils.data as data
from utils import *
from scipy import stats

class sperfloader(data.Dataset):
    """
    dataloader for loading *total* projection data
    """
    def __init__(self, projdir = '.', 
                       psfdir = '.',
                       encoder_level = 10, 
                       downfactor = 2, 
                       start = 0,
                       norm_type = 'none', 
                       encode_type='exponential',
                       mode = 'train',
                       debug = False):
        
        super(sperfloader, self).__init__()
        assert start < downfactor
        # convert projections from jld2 type to numpy array
        self.proj = convertjld2numpy(projdir)
        if debug:
            self.proj = self.proj[:,:-1,:]
        self.nx, self.nz, self.nview = self.proj.shape
        print('max of proj: ', np.amax(self.proj))
        # psf files are radial position of cameras, having unit in mm.
        self.psf = np.loadtxt(psfdir) / 300 # Isaac: why 300?
        self.mode = mode
        # pre-processing
        if norm_type == 'none':
            self.min_val =  0
            self.proj = self.proj - self.min_val
            self.max_val  = 1
            self.proj = self.proj / self.max_val
        elif norm_type  == 'minmax':
            self.min_val =  np.min(self.proj)
            self.proj = self.proj - self.min_val
            self.max_val  = np.std(self.proj)#np.max(self.proj)
            self.proj = self.proj / self.max_val
        elif norm_type  == 'std':
            self.min_val =  np.mean(self.proj)#np.min(self.proj)
            self.max_val  = np.std(self.proj)#np.max(self.proj)
            self.proj = self.proj - self.min_val
            self.proj = self.proj / self.max_val
        else:
            print(f'normalization method = {norm_type} not found !')
            exit(0)
        
        # Isaac: For reducing memory:
        self.proj = self.proj.astype(np.float32)

        print('projections shape: ', self.proj.shape)
        self.reduced_view = int(self.nview / downfactor)
        print(f'reduced from {self.nview} views to {self.reduced_view} views')
        self.totlen = self.nx * self.nz * self.reduced_view
        self.input_grid = self.gen_grid()
        print('grid size is: ', self.input_grid.shape) # (4, nz, nx, nview, nview) (x, y, sin, cos) # Isaac: (z, x, cos, sin)?
        print('memory taken by the grid is: {:.2f}'.format(self.input_grid.itemsize*self.input_grid.size/1024/1024/1024))
        
        # (1, 4, nz, nx, nview, nview) -> (nx, nz, nview, nview, 4, 1)
        self.input_grid_encoded = np.array([self.input_grid]).transpose((3, 2, 4, 5, 1, 0))
           
        print('all grid shape: ', self.input_grid_encoded.shape)
        train_view_index = np.zeros(self.nview, dtype=bool)
        train_view_index[np.arange(start, self.nview, downfactor)] = True
        test_view_index = np.ones(self.nview, dtype=bool)
        test_view_index[np.arange(start, self.nview, downfactor)] = False
        
        self.train_psf = self.psf[train_view_index]
        print('train psf size: ', self.train_psf.shape) # (reduced_view,)
        self.test_psf = self.gen_interp_psf(self.train_psf, train_view_index, test_view_index) # linear interpolated psf (radial positions)
        
        self.train_psf_grid = self.gen_psf_grid(self.train_psf).astype(np.float32)
        print('train psf grid size: ', self.train_psf_grid.shape) # (nx, nz, reduced_view, 1, 1)
        self.test_psf_grid = self.gen_psf_grid(self.test_psf).astype(np.float32)
        
        self.train_grid_encoded = self.input_grid_encoded[:, :, train_view_index, train_view_index, :, :]
        # (nx, nz, reduced_view, 4, 1) 
        
        self.test_grid_encoded = self.input_grid_encoded[:, :, test_view_index, test_view_index, :, :]
        # (nx, nz, nview - reduced_view, 4, 1)

        self.train_grid_encoded = np.concatenate((self.train_grid_encoded, self.train_psf_grid), axis = 3)
        # (nx, nz, reduced_view, 5, 1)
        
        self.test_grid_encoded = np.concatenate((self.test_grid_encoded, self.test_psf_grid), axis = 3)
        # (nx, nz, nview - reduced_view, 5, 1)
        
        print("train grid size before encoding:", self.train_grid_encoded.shape) 
        print("test grid size before encoding:", self.test_grid_encoded.shape) 
        
        if encoder_level > 0:
            self.train_grid_sin = np.array([gen_sin(self.train_grid_encoded, i, pe=encode_type) for i in range(encoder_level)])
            # (encoder level, nx, nz, reduced_view, 5, 1)
            self.train_grid_cos = np.array([gen_cos(self.train_grid_encoded, i, pe=encode_type) for i in range(encoder_level)])
            # (encoder level, nx, nz, reduced_view, 5, 1)
            self.train_grid_cat = np.concatenate((self.train_grid_sin, self.train_grid_cos), axis=0)
            # (2*encoder level, nx, nz, reduced_view, 5, 1) 
            
            self.test_grid_sin = np.array([gen_sin(self.test_grid_encoded, i, pe=encode_type) for i in range(encoder_level)])
            # (encoder level, nx, nz, nview-reduced_view, 5, 1)
            self.test_grid_cos = np.array([gen_cos(self.test_grid_encoded, i, pe=encode_type) for i in range(encoder_level)])
            # (encoder level, nx, nz, nview-reduced_view, 5, 1)
            self.test_grid_cat = np.concatenate((self.test_grid_sin, self.test_grid_cos), axis=0)
            # (2*encoder level, nx, nz, nview-reduced_view, 5, 1) 
            
            self.train_grid_cat = self.train_grid_cat.transpose((1,2,3,0,4,5)).reshape(self.nx, self.nz, self.reduced_view, -1, 1) 
            # (2*encoder level, nx, nz, reduced_view, 5, 1) -> (nx, nz, reduced_view, 10*encoder level, 1)
            
            self.test_grid_cat = self.test_grid_cat.transpose((1,2,3,0,4,5)).reshape(self.nx, self.nz, self.nview-self.reduced_view, -1, 1) 
            # (2*encoder level, nx, nz, nview-reduced_view, 5, 1) -> (nx, nz, nview-reduced_view, 10*encoder level, 1)
            
            self.train_grid_encoded = np.concatenate((self.train_grid_encoded,
                                                      self.train_grid_cat),
                                                     axis=3)
            self.test_grid_encoded = np.concatenate((self.test_grid_encoded,
                                                      self.test_grid_cat),
                                                     axis=3)
            # (nx, nz, nview, 10*encoder level, 1) -> (nx, nz, nview, 10*encoder level + 5, 1)
            
        print("train grid size after encoding:", self.train_grid_encoded.shape) 
        print("test grid size after encoding:", self.test_grid_encoded.shape) 
        
        self.reduced_proj_train = self.proj[:, :, train_view_index]  # (nx, nz, reduced_view)
        self.reduced_proj_test = self.proj[:, :, test_view_index] # (nx, nz, nview - reduced_view)
        print("train proj values size:", self.reduced_proj_train.shape) # (nx, nz, reduced_view)
        print("test proj values size:", self.reduced_proj_test.shape)  # (nx, nz, nview-reduced_view)
        param_len = self.train_grid_encoded.shape[3] # 8*encoder level+5
        
        self.reshaped_grid_train = self.train_grid_encoded.reshape(-1, param_len) # 2D matrix: (nx*nz*reduced_view, 5)
        self.reshaped_grid_test = self.test_grid_encoded.reshape(-1, param_len) # 2D matrix: (nx*nz*(nview-reduced_view), 5)
            
        self.reshaped_proj_train = self.reduced_proj_train.reshape(-1) 
        # self.reshaped_proj_valid = self.reduced_proj_train.reshape(-1)[valid_idx]
        self.reshaped_proj_test = self.reduced_proj_test.reshape(-1)
        print("reshaped train grid size:", self.reshaped_grid_train.shape)  # (nx*nz*reduced_view, 5)
        print("reshaped test grid size:", self.reshaped_grid_test.shape)  # (nx*nz*(nview-reduced_view), 5)
        print("reshaped train proj values size:", self.reshaped_proj_train.shape)  # (nx*nz*reduced_view, )
        print("reshaped test proj values size:", self.reshaped_proj_test.shape) # (nx*nz*(nview-reduced_view), )

    def __len__(self):
        if self.mode == 'train':
            return self.nx * self.nz * self.reduced_view
        else:
            return self.nx * self.nz * (self.nview - self.reduced_view)

    def gen_psf_grid(self, x): # (nview,)
        y = np.expand_dims(x, axis=(0, 1)) # (1, 1, nview)
        z1 = np.repeat(y, self.nx, axis = 0) # (nx, 1, nview)
        z2 = np.repeat(z1, self.nz, axis = 1) # (nx, nz, nview)
        z3 = np.expand_dims(z2, axis = -1) # (nx, nz, nview, 1)
        return np.expand_dims(z3, axis = -1) # (nx, nz, nview, 1, 1)
    
    def gen_interp_psf(self, train_psf, train_view_index, test_view_index):
        all_idx = np.arange(self.nview)
        train_idx = all_idx[train_view_index]
        test_idx = all_idx[test_view_index]    
        test_psf = np.interp(test_idx, train_idx, train_psf)
        return test_psf
    
    def getsize(self):
        return self.nx, self.nz, self.reduced_view, self.nview - self.reduced_view # of views for testing
    
    def __getitem__(self, item):
        Meta = {}
        if self.mode == 'train':
            Meta['train_grid'] = torch.from_numpy(self.reshaped_grid_train[item, :]).to(torch.float)
            Meta['train_proj'] = torch.from_numpy(np.asarray(self.reshaped_proj_train[item])).to(torch.float).unsqueeze(-1)
            return Meta
        else:
            Meta['test_grid'] = torch.from_numpy(self.reshaped_grid_test[item, :]).to(torch.float)
            Meta['test_proj'] = torch.from_numpy(np.asarray(self.reshaped_proj_test[item])).to(torch.float).unsqueeze(-1)
            return Meta

    def gen_grid(self):
        # Isaac: for reducing memory
        xgrid = np.linspace(-1, 1, self.nx).astype(np.float32)
        zgrid = np.linspace(-1, 1, self.nz).astype(np.float32)
        viewgrid = np.linspace(0, 2*np.pi, self.nview).astype(np.float32)
        viewgrid_cos = np.cos(viewgrid)
        viewgrid_sin = np.sin(viewgrid)
        input_grid = np.stack(np.meshgrid(xgrid, zgrid, viewgrid_cos, viewgrid_sin))
        return input_grid.astype(np.float32)

if __name__ == "__main__":
    # background and total projections are all noisy

    proj_path = "../patient_data/proj/proj_patient4_c1_s1.jld2"
    psf_path = "../patient_data/radial_position/radial_position_proj_patient4_c1_s1.txt"
    loader = sperfloader(projdir = proj_path, psfdir = psf_path, encoder_level = 0, downfactor = 4, mode = 'train')
    print('length of loader: ', len(loader))
    idx = 314000
    print('a sample from loader: ', loader[idx])



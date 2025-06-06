# utils.py
# useful functions
import h5py
import numpy as np
import matplotlib.pyplot as plt
import configargparse
import os
import torch
import shutil
import random

def convertjld2numpy(filename):
    """
    :param filename: needs to be a jld
    :return: .npy
    """
    f = h5py.File(filename, "r")
    # numpy reads in arrays reverse dimension from julia so flip the dims
    x = f[list(f.keys())[-1]][()].transpose((2,1,0))
    return x

def positonal_encoder(x, L=10):
    """
    :param x: coordinate value
    :param L: order number
    :return: g = (sin(2^0 pi x), cos(2^0 pi x), ..., sin(2^(L-1) pi x), cos(2^(L-1) pi x))
    """
    g = np.zeros(2*L)
    for i in range(L):
        g[2*i] = np.sin((2**i) * np.pi * x)
        g[2*i+1] = np.cos((2**i) * np.pi * x)
    return g

def gen_sin(x, i, pe='exponential'):
    if pe == 'exponential':
        encode = np.sin((2**i) * np.pi * x)
    elif pe == 'linear':
        encode = np.sin(0.5 * (i + 1) * np.pi * x)
    return encode

def gen_cos(x, i, pe='exponential'):
    if pe == 'exponential':
        encode = np.cos((2**i) * np.pi * x)
    elif pe == 'linear':
        encode = np.cos(0.5 * (i + 1) * np.pi * x)
    return encode

def laplace(y, x):
    grad = gradient(y, x)
    return divergence(grad, x)


def divergence(y, x):
    div = 0.
    for i in range(y.shape[-1]):
        div += torch.autograd.grad(y[..., i], x, torch.ones_like(y[..., i]), create_graph=True)[0][..., i:i+1]
    return div


def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad

def config_parser():
    # Isaac: The parser will automatically search for and read from the file specified by the path
    # **Missing Fields**: If `params.txt` does not contain all the fields that your script expects, 
    # the parser will use the default values specified in your `add_argument` calls for those missing fields. 
    # If no default value is specified in the script and the field is not in `params.txt`, an error may occur, 
    # especially if the field is required.
    # **Extra Fields**: If your `params.txt` file contains additional fields that are not defined in your script 
    # with `add_argument`, those fields will generally be ignored by the parser. 
    # `configargparse` focuses on the arguments you've defined in your script.

    parser = configargparse.ArgumentParser(default_config_files=['../config/params.txt'])
    parser.add_argument('--config', is_config_file=True,
                        help='config file path')
    parser.add_argument("--expname", type=str, help='experiment name')
    parser.add_argument("--logsdir", type=str, default='../logs/',
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='../data/',
                        help='where to load training/testing data')
    parser.add_argument("--psfdir", type=str, default = '../psf/',
                        help='place to load psf files')

    parser.add_argument("--netdepth", type=int, default=8,
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256,
                        help='channels per layer')
    parser.add_argument("--encoderlevel", type=int, default=10,
                        help='levels of positional encoder')

    parser.add_argument("--nepochs", type=int, default=200,
                        help='number of epochs for training')

    parser.add_argument("--lr", type=float, default=0.001,
                        help='learning rate')
    parser.add_argument("--optim", type=str, default='sgd',
                        help='optimizer')
    parser.add_argument("--batchsize", type=int, default=1000,
                        help='batch size')
    parser.add_argument("--validfrac", type=float, default=0.2,
                        help='fraction of validation data')
    parser.add_argument("--downfactor", type=int, default=2,
                        help='downsample factor')
    parser.add_argument("--start", type=int, default=0,
                        help='start view angle')
    parser.add_argument("--model", type=str, default='nerf',
                        help='network architecture')
    # Isaac: What's these two paras?
    parser.add_argument("--firstomega", type=float, default=3,
                        help='first omega 0 for siren')
    parser.add_argument("--hiddenomega", type=float, default=3,
                        help='hidden omega 0 for siren')

    parser.add_argument("--do_online_test", type=bool, default=False,
                        help='if do testing while training')

    parser.add_argument("--online_test_epoch_gap", type=int, default=1,
                        help='the gap between each online test')
    
    parser.add_argument("--loss_fun", type=str, default='l2',
                        help='training loss')
    
    parser.add_argument("--gpu_ids", type=str, default='0',
                        help='the index of the gpu to use')

    parser.add_argument("--srcdir", type=str, default='../src/',
                        help='where the code is')

    return parser

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_checkpoint(state, is_best, checkpoint_dir, logger=None):
    """Saves model and training parameters at '{checkpoint_dir}/last_checkpoint.pytorch'.
    If is_best==True saves '{checkpoint_dir}/best_checkpoint.pytorch' as well.
    Args:
        state (dict): contains model's state_dict, optimizer's state_dict, epoch
            and best evaluation metric value so far
        is_best (bool): if True state contains the best model seen so far
        checkpoint_dir (string): directory where the checkpoint are to be saved
    """

    def log_info(message):
        if logger is not None:
            logger.info(message)

    if not os.path.exists(checkpoint_dir):
        log_info(
            f"Checkpoint directory does not exists. Creating {checkpoint_dir}")
        os.mkdir(checkpoint_dir)

    last_file_path = os.path.join(checkpoint_dir, 'last_checkpoint.pytorch')
    log_info(f"Saving last checkpoint to '{last_file_path}'")
    torch.save(state, last_file_path)
    if is_best:
        best_file_path = os.path.join(checkpoint_dir, 'best_checkpoint.pytorch')
        log_info(f"Saving best checkpoint to '{best_file_path}'")
        shutil.copyfile(last_file_path, best_file_path)
        
###############################################################################
# Make folders
###############################################################################

def check_and_mkdir(path):
    if not os.path.exists(path):
        # os.mkdir(path)
        os.makedirs(path)
        
def computeSNR(x, xhat): 
    return 20 * torch.log10(torch.norm(x.flatten())/torch.norm(x.flatten()-xhat.flatten()))
        
###############################################################################
# Init envs
###############################################################################
def init_env(seed_value=42):
    os.environ['PYTHONHASHSEED']=str(seed_value)
    random.seed(seed_value)
    torch.manual_seed(seed_value)
    np.random.seed(seed_value)

###############################################################################
# Backup codes
###############################################################################

def copytree_code(src_path, save_path):
    max_code_save = 100
    for i in range(max_code_save):
        code_path = save_path + 'code%d/' % i
        if not os.path.exists(code_path):
            shutil.copytree(src=src_path, dst=code_path)
            break
    
if __name__ == "__main__":
    # background and total projections are all noisy
    # bg_path = "../lu177data/vp6/background.jld"
    # tot_path = "../lu177data/vp6/noisy.jld"
    # background = convertjld2numpy(bg_path)
    # ynoisy = convertjld2numpy(tot_path)
    # print("maximum of background: ", np.max(background))
    # print("minimum of background: ", np.min(background))
    # print("maximum of projection: ", np.max(ynoisy))
    # print("minimum of projection: ", np.min(ynoisy))
    # plt.figure()
    # # plt.imshow(background[:,:,0].T)
    # plt.imshow(ynoisy[:,:,30].T)
    # plt.show()
    # p = 0.4141
    # L = 10
    # g = positonal_encoder(p, L)
    # print(g)
    parser = config_parser()
    args = parser.parse_args()
    print(args)



# main.py
# main file to run all codes
from train import sperftrainer
import torch.utils.tensorboard # for launching tensorboard
from test import sperftester
import numpy as np
from utils import *
import scipy.io as sio

if __name__ == "__main__":
    trainer = sperftrainer()
    trainer.train()
    checkpoint_dir = trainer.expdir
    tester = sperftester(f'{checkpoint_dir}/models', dataset_key='test')
    results, GT = tester.test(dataset_key = 'test', denormalize = True)
    train_tester = sperftester(f'{checkpoint_dir}/models', dataset_key='train')
    results_train, _ = train_tester.test(dataset_key = 'train', denormalize = True)
    train_proj = tester.loader.reduced_proj_train
    nx, nz, ntrainview, ntestview = tester.nx, tester.nz, train_tester.trainview, tester.testview
    print(f'ntrainview: {ntrainview}, ntestview: {ntestview}')
    results = results.reshape(nx, nz, ntestview)
    GT = GT.reshape(nx, nz, ntestview)
    results_train = results_train.reshape(nx, nz, ntrainview)
    
    test_savedir = f'{checkpoint_dir}/test'
    check_and_mkdir(test_savedir)
    # np.save(f'{test_savedir}/data.npy', 
    #         {'pre':results,'gdt': GT})
    np.save(f'{test_savedir}/pred_proj_test.npy', results)
    np.save(f'{test_savedir}/gt_proj_test.npy', GT)
    np.save(f'{test_savedir}/gt_proj_train.npy', train_proj)
    np.save(f'{test_savedir}/pred_proj_train.npy', results_train)
    
    sio.savemat(f'{test_savedir}/pred_proj_test.mat', {'results': results})
    sio.savemat(f'{test_savedir}/gt_proj_test.mat', {'GT': GT})
    sio.savemat(f'{test_savedir}/gt_proj_train.mat', {'train_proj': train_proj})
    sio.savemat(f'{test_savedir}/pred_proj_train.mat', {'results_train': results_train})
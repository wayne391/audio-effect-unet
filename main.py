'''
audio effect
'''

import os
import json
import argparse
import numpy as np

import solver
from models.m_unet import Net
from dataset import AudioEffectDataset

import torch
from config import DEVICE, SR

import librosa
import librosa.display
import matplotlib.pyplot as plt


print('device:', DEVICE)
parser = argparse.ArgumentParser(description='drum transcription')


# training
parser.add_argument('--device', type=str, default=DEVICE,  help='number of conv feature maps (F)')
parser.add_argument('--batch_size', type=int, default=32, help='number of color channels to use')
parser.add_argument('--epochs', type=int, default=1, help='number of residual blocks (NB)')
parser.add_argument('--lr', type=float, default=5*1e-5, help='learning rate')
parser.add_argument('--step_size', type=float, default=10000, help='LRstep size')
parser.add_argument('--gamma', type=float, default=0.5, help='LRstep ratio')


# monitoring
parser.add_argument('--interval_log', type=int, default=2, help='number of residual groups (NG)')
parser.add_argument('--interval_val', type=int, default=4, help='number of feature maps reduction')
parser.add_argument('--interval_ckpt', type=int, default=4, help='number of feature maps reduction')


# Saver
parser.add_argument('--exp_dir', default='exp', help='datasave directory')

args = parser.parse_args()


def main():
    # load data
    path_json = 'data/file_pair.json'
    with open(path_json, 'r') as f:
        pair_list = json.load(f)

    path_npz_train = os.path.join('data', 'processed_train.npz')
    path_npz_test = os.path.join('data', 'processed_test.npz')

    # create datasets
    train_set = AudioEffectDataset(pair_list['train'], path_npz_train, DEVICE)
    test_set = AudioEffectDataset(pair_list['test'], path_npz_test, DEVICE)

    # model
    model = Net()
    
    # training
    solver.train(args, model, train_set, valid_set=test_set)

    # testing
    print('[*] testing...')
    model = solver.restore(args.exp_dir)
    test_set.return_mode = 1
    _, loss = solver.test(
        args, 
        model, 
        test_set, 
        is_report=True, 
        is_shuffle=True, 
        outdir=os.path.join(args.exp_dir, 'test'))
    print('loss:', np.mean(loss))


if __name__ == '__main__':
    main()


# save config
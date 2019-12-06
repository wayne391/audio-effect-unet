'''
audio effect
'''

import os
import sys
import glob
import json
import collections
import numpy as np

import librosa
import soundfile as sf
# import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset

from config import SR, WIN_LEN, HOP_LEN


class AudioEffectDataset(Dataset):
    def __init__(self, pair_list, path_npz, device, is_seg=True):
        self.device = device
        self.is_seg = is_seg

        # laod processed data
        if os.path.exists(path_npz):
            print('[o] processed data exists')
            loaded = np.load(path_npz, allow_pickle=True)
            self.pair_list = loaded['data']
            self.num_file = len(pair_list)
            return 

        # no processed data, run from scratch
        self.pair_list = pair_list
        self.num_file = len(pair_list)

        print('\n\nnumber of files:', self.num_file)
        print('[*] dataset loading...')

        # load audio
        for idx, p in enumerate(pair_list):
            sys.stdout.write('{}/{} \r'.format(idx, self.num_file))
            sys.stdout.flush()
            p['x'], _ = librosa.load(p['x'], sr=SR)
            p['y'], _ = librosa.load(p['y'], sr=SR)

            p['x'] = p['x']
            p['y'] = p['y']

        # save
        np.savez(path_npz, data=self.pair_list)
        print('num of train:', self.num_file)

    def __getitem__(self, index):
        p = self.pair_list[index]
        audio_len = min(len(p['x']), len(p['y']))

        result = dict()
        if self.is_seg:
            # segment audio by WIN_LEN
            sample_begin = np.random.randint(audio_len - WIN_LEN - 1)
            sample_end = sample_begin + WIN_LEN

            result = {
                'x': torch.from_numpy(p['x'][sample_begin:sample_end][np.newaxis, ...]).float().to(self.device),
                'y': torch.from_numpy(p['y'][sample_begin:sample_end][np.newaxis, ...]).float().to(self.device),
                'setting': p['setting'],
            }
        else:
            result = {
                'x': torch.from_numpy(p['x']),
                'y': torch.from_numpy(p['y']),
                'setting': p['setting'],
            }

        return result

    def __len__(self):
        return self.num_file


def create_file_pair(
        dir_noFX, 
        dir_effect, 
        effect_str, 
        path_file,
        ext='wav'):

    list_noFX = glob.glob(os.path.join(dir_noFX, '*.' + ext))
    list_effect = glob.glob(os.path.join(dir_effect, '*.'+ ext))

    print('\n'+'='*40)
    print(' > files in noFX:  ', len(list_noFX))
    print(' > files in effect:', len(list_effect))

    pair_dict = collections.defaultdict(dict)

    # noFX
    for _, file in enumerate(list_noFX):
        fn = os.path.basename(file)
        unique_key = fn[:9]
        pair_dict[unique_key]['x'] = file

    # effect
    for _, file in enumerate(list_effect):
        fn = os.path.basename(file)
        unique_key = fn[:9]
        pair_dict[unique_key]['y'] = file
        
    # sanity check
    pair_list = []
    for k, v in pair_dict.items():
        entry = {
            'setting': k,
            'x': v['x'],
            'y': v['y'],
        }
        pair_list.append(entry )
    pair_list.sort(key=lambda x: x['setting'])
    print(' > umber of pairs:', len(pair_list)) 
    print(' > Sample:')
    for k, v in pair_list[0].items():
        print('    {}:{}'.format(k, v))
    print('='*40)

    # train/test split
    num_pair = len(pair_list)
    ratio_test = 0.1
    rand_order = np.random.permutation(num_pair) 
    test_idx = rand_order[:int(num_pair * ratio_test)]
    train_idx = rand_order[int(num_pair * ratio_test):]

    # output
    final = {
        'train': [],
        'test': [],
    }
    for idx in train_idx:
        final['train'].append(pair_list[idx])

    for idx in test_idx:
        final['test'].append(pair_list[idx])

    with open(path_file, 'w') as f:
        json.dump(final, f)
    return final


def create_training_set(
        pair_list,
        path_samples,
        sr = 16000,
        win_len=1024,
        hop_len=64):

    cnt = 0
    ## fidx = 20
    for fidx in range(len(pair_list)):            
        print(' > ', fidx)
        pair = pair_list[fidx]
        file_x = pair['x']
        file_y = pair['y']

        # wav_x, sr = sf.read(file_x)
        # wav_y, sr = sf.read(file_y)
        
        wav_x, _ = librosa.load(file_x, sr=sr)
        wav_y, _ = librosa.load(file_y, sr=sr)
        print('wav_x:', wav_x.shape)
        print('wav_y:', wav_y.shape)

        for i in range(0, len(wav_x), hop_len):
            st = i 
            ed = i + win_len
        
            fn = '{}_{}_{}.npz'.format(pair['setting'], st, ed)
            path_out = os.path.join(path_samples, fn)
            np.savez(
                path_out, 
                x=wav_x[st:ed],
                y=wav_y[st:ed])
            
            cnt += 1
            
    print('num of samples:', cnt)


if __name__ == '__main__':
    # config - input
    root = os.path.join('Gitarre-monophon', 'Samples')
    dir_noFX = os.path.join(root, 'NoFX')
    dir_effect = os.path.join(root, 'Distortion')
    effect_str = '4411' 
    path_json = 'data/file_pair.json'

    # ----------------- #
    #  create data pair #
    # ----------------- #

    # config - output
    path_samples = os.path.join('data', 'samples')

    # creat file pair in json
    _ = create_file_pair(
        dir_noFX, 
        dir_effect, 
        effect_str,
        path_json)
    
    with open(path_json, 'r') as f:
        pair_list = json.load(f)

    # create file pair (too big)
    # create_training_set(
    #     pair_list,
    #     path_samples,
    #     win_len=1024,
    #     hop_len=64)

    # ---------------- #
    #  build dataset   #
    # ---------------- #

    # load data
    path_npz_train = os.path.join('data', 'processed_train.npz')
    path_npz_test = os.path.join('data', 'processed_test.npz')
    train_set = AudioEffectDataset(pair_list['train'], path_npz_train, 0)
    test_set = AudioEffectDataset(pair_list['test'], path_npz_test, 0)

    # display to check
    print('\n\n[*] check dataset')
    print(' -- train set -- ')
    print(' > num files:', len(train_set))

    idx = 40
    print('    x: ', train_set[idx]['x'].shape)
    print('    y: ', train_set[idx]['y'].shape)
    print('    setting: ', train_set[idx]['setting'])
    print(max(train_set[idx]['x']), min(train_set[idx]['x']))

    print(' -- test set -- ')
    print(' > num files:', len(test_set))

    idx = 0
    print('    x: ', test_set[idx]['x'].shape)
    print('    y: ', test_set[idx]['y'].shape)
    print('    setting: ', test_set[idx]['setting'])



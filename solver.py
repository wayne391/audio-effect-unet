'''
audio effect
'''

import os
import time
import datetime
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR

from config import DEVICE, WIN_LEN, SR
from dataset import restore_cropped_song, crop_song
from report import make_loss_report, make_song_report

import warnings
warnings.filterwarnings("ignore")


def restore(path_exp, name= 'model.pt'):
    path_pt = os.path.join(path_exp, name)
    model = torch.load(path_pt, map_location=torch.device(DEVICE))
    return model


def _get_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def _tensor_to_numpy(tensor):
    if tensor.requires_grad:
        tensor = tensor.detach()
    if DEVICE == 'cuda':
        tensor = tensor.cpu()
    return tensor.numpy()


def network_paras(model):
    # compute only trainable params
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


def run_on_song(model, song, is_report=False, outdir='.', prefix=''):
    # fetch
    wav_x = song['x']
    wav_y = song['y']

    if wav_x.shape[0] == 1:
        wav_x = wav_x.squeeze(0)
    if wav_y.shape[0] == 1:
        wav_y = wav_y.squeeze(0)

    # crop
    song_batch_x = crop_song(wav_x, WIN_LEN)
    song_batch_x = np.expand_dims(song_batch_x, axis=1)
    song_batch_y = crop_song(wav_y, WIN_LEN)
    song_batch_y = np.expand_dims(song_batch_y, axis=1)

    batch = {
        'x': torch.from_numpy(song_batch_x),
        'y': torch.from_numpy(song_batch_y),
    }

    # run song batch
    batch_y_pred, loss = run_on_batch(model, batch)

    # restore
    wav_y_pred = restore_cropped_song(batch_y_pred)

    # report
    if is_report:
        if outdir:
            if not os.path.exists(outdir):
                os.makedirs(outdir)
        wav_x = _tensor_to_numpy(wav_x)
        wav_y = _tensor_to_numpy(wav_y)
        make_song_report(wav_x, wav_y, wav_y_pred, outdir, prefix=prefix)

    return wav_y_pred, loss


def run_on_batch(model, batch):
    # fetch
    wav_x = batch['x']
    wav_y = batch['y']

    # sanity check
    # print('model: ', wav_x.size()) # 0: (32, 1, 1024)
    # print('model: ', wav_y.size()) # 0: (32, 1, 1024)

    # run model
    wav_y_pred = model(wav_x)

    # squeeze channel
    wav_y_pred = torch.squeeze(wav_y_pred)

    # compute loss
    if wav_y is not None:
        wav_y = torch.squeeze(wav_y)
        loss = model.compute_loss(wav_y, wav_y_pred)
    else:
        loss = None
    return wav_y_pred, loss


def test(
        args, 
        model, 
        test_set, 
        amount=None,
        is_shuffle=False,
        is_report=False,
        outdir=None, 
        prefix=''):

    model.eval()

    # config
    mode = test_set.return_mode
    print(' > (testing) mode:', mode)
    if mode == 0:
        batch_size = args.batch_size
    elif mode == 1:
        batch_size = 1
    else:
        raise ValueError('Unacceptable mode')
    
    # dataloader
    test_loader = DataLoader(test_set, batch_size, shuffle=is_shuffle, drop_last=False)

    # init
    score_all = []
    y_pred_list = []

    # run
    for idx, batch in enumerate(test_loader):
        if amount and idx >= amount:
            break

        # forward
        if mode == 0:
            y_pred, loss = run_on_batch(model, batch)
        elif mode == 1:  
            y_pred, loss = run_on_song(
                model, 
                batch, 
                is_report=is_report, 
                outdir=outdir, 
                prefix=prefix+str(idx)+'-')
        else:
            raise ValueError('Unacceptable mode')

        # to numpy
        if isinstance(y_pred, torch.Tensor):
            y_pred = _tensor_to_numpy(y_pred)

        # add to list
        y_pred_list.append(y_pred)
        score_all.append(loss.item())

    return y_pred_list, score_all


def train(args, model, train_set, valid_set=None):
    amount = network_paras(model)
    print('params amount:', amount)

    # create exp folder
    if not os.path.exists(args.exp_dir):
        os.makedirs(args.exp_dir)
    path_log = os.path.join(args.exp_dir, 'log.txt')
    with open(path_log, 'w') as fp:
        fp.write('\n')

    # config
    model.train()
    model.to(args.device)

    # data
    dataloader = DataLoader(train_set, args.batch_size, shuffle=True, drop_last=True)

    # training config
    counter = 0
    is_valid = True if valid_set is not None else False
    num_batch = len(train_set) // args.batch_size
    acc_batch_time = 0
    time_start_train = time.time()

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    scheduler = StepLR(optimizer, step_size=args.step_size*num_batch, gamma=args.gamma)

    # training phase
    print('{:=^40}'.format(' start training '))
    for epoch in range(args.epochs):
        for bidx, batch in enumerate(dataloader):
            time_start_batch = time.time()
            counter += 1 

            # forwarding
            _, loss = run_on_batch(model, batch)

            # update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            # monitoring
            ## print loss
            if counter % args.interval_log == 0: 
                acc_batch_time += time.time() - time_start_batch
                train_time = time.time() - time_start_train
                
                log = 'epoch: %d/%d (%3d/%3d) | %s | t: %.2f | loss: %.6f | time: %s | counter: %d' % (
                    epoch, 
                    args.epochs, 
                    bidx, num_batch, 
                    args.exp_dir,
                    acc_batch_time,
                    loss.item(), 
                    str(datetime.timedelta(seconds=train_time))[:-5],
                    counter)

                print(log)
                with open(path_log, 'a') as fp:
                    fp.write(log + '\n')
                acc_batch_time = 0

            ## validation
            if counter % args.interval_val == 0 and is_valid:
                print(' [*] run validation...')
                model.eval()

                valid_set.return_mode = 0
                _, scores = test(args, model, valid_set)
                lr = _get_learning_rate(optimizer)
                log = ' > validation loss: %.6f | lr: %.6f | counter: %d' % (np.mean(scores), lr, counter)

                print(log)
                with open(path_log, 'a') as fp:
                    fp.write(log + '\n')
                
                model.train()

            ## saving
            if counter % args.interval_ckpt == 0:
                # save model
                print(' [*] saving model...')
                torch.save(model, os.path.join(args.exp_dir, 'model.pt'))
                torch.save(optimizer.state_dict(), os.path.join(args.exp_dir, 'optimizer.pt'))

                # save random result
                print(' [*] generating sanpshots...')
                valid_set.return_mode = 1
                test(
                    args, 
                    model, 
                    valid_set, 
                    amount=3, 
                    is_shuffle=True, 
                    is_report=True, 
                    outdir=os.path.join(args.exp_dir, 'sample'),
                    prefix=str(counter)+'-')
                
                # make report
                make_loss_report(path_log, os.path.join(args.exp_dir, 'loss_report.png'))

    # done
    print('{:=^40}'.format(' Finished '))

    # save
    torch.save(model, os.path.join(args.exp_dir, 'model.pt'))
    torch.save(optimizer.state_dict(), os.path.join(args.exp_dir, 'optimizer.pt'))
    make_loss_report(path_log, os.path.join(args.exp_dir, 'loss_report.png'))

    # runtime
    runtime = time.time() - time_start_train
    print('training time:', str(datetime.timedelta(seconds=runtime))+'\n\n')

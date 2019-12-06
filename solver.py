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

import warnings
warnings.filterwarnings("ignore")


device = 'cuda' if torch.cuda.is_available() else 'cpu'


def restore(path_exp, name= 'model.pt'):
    path_pt = os.path.join(path_exp, name)
    model = torch.load(path_pt, map_location=torch.device(device))
    return model



def _get_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def network_paras(model):
    # compute only trainable params
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params
    

def test(args, model, test_set, is_eval=False):
    model.eval()
    test_loader = DataLoader(test_set, 1, shuffle=True, drop_last=True)
    score_all = []
    y_pred_list = []
    for _, batch in enumerate(test_loader):
        y_pred, loss = model.run_on_batch(batch)
        if device == 'cuda':
            y_pred = y_pred.cpu()
        y_pred = y_pred.detach().numpy()
        y_pred_list.append(y_pred)
        if is_eval:
            score_all.append(loss.item())
    if is_eval:
        return y_pred_list, np.array(score_all)
    else:
        return y_pred_list


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
            _, loss = model.run_on_batch(batch)

            # update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            acc_batch_time += time.time() - time_start_batch

            # monitoring
            ## print loss
            if counter % args.interval_log == 0: 
                lr = _get_learning_rate(optimizer)
                log = 'epoch: (%d/%d) [%5d/%5d], loss: %.6f | lr: %.6f | time: %s' % \
                    (epoch, args.epochs, bidx, num_batch, loss.item(), lr, str(datetime.timedelta(seconds=acc_batch_time)))
                print(log)
                with open(path_log, 'a') as fp:
                    fp.write(log + '\n')
                acc_batch_time = 0

            ## validation
            if counter % args.interval_val == 0 and is_valid:
                print(' [*] run validation...')
                model.eval()
                _, scores = test(args, model, valid_set, is_eval=True)
                log = ' > validation loss:', np.mean(scores)
                print(log)
                with open(path_log, 'a') as fp:
                    fp.write(log + '\n')
                
                model.train()

            ## saving
            if counter % args.interval_ckpt == 0:
                print(' [*] ckpt. Saving Model...')
                torch.save(model, os.path.join(args.exp_dir, 'model.pt'))
                torch.save(optimizer.state_dict(), os.path.join(args.exp_dir, 'optimizer.pt'))
    

    # done
    print('{:=^40}'.format(' Finished '))

    # save
    torch.save(model, os.path.join(args.exp_dir, 'model.pt'))
    torch.save(optimizer.state_dict(), os.path.join(args.exp_dir, 'optimizer.pt'))

    # runtime
    runtime = time.time() - time_start_train
    print('training time:', str(datetime.timedelta(seconds=runtime))+'\n\n')

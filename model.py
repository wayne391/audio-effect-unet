'''
audio effect
'''

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv(in_channels, out_channels, kernel_size, bias=False):
    return nn.Conv1d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        padding=(kernel_size-1)//2,  # same padding
        bias=bias)


def linear_softplus(in_feature, out_feature):
    return nn.Sequential(
        nn.Linear(in_feature, out_feature),
        nn.Softplus()
    )
    

class Net(nn.Module):
    def __init__(self):
        '''
        all convolutions are along the time dime
        stride = 1 (move the filters one sample at a time)
        '''

        super(Net, self).__init__()
        # Conv1d: in_channels, out_channels, kernel_size

        # Adaptive Front-end
        self.conv1_e1 = conv(1, 128, 63)
        self.conv2_e2 = conv(128, 128, 127)
        self.conv_act = nn.Softplus()

        # bn?
        self.maxpool = nn.MaxPool1d(16, return_indices=True)

        #  bottleneck
        self.linear_z1 = linear_softplus(64, 64) # 64, softplus
        self.linear_z2 = linear_softplus(64, 64) # 64, softplus

        # Synthesis Back-end
        self.unpool = nn.MaxUnpool1d(16)

        ## DNN-SAAF
        self.linear_d1 = linear_softplus(128, 128) # 128, softplus
        self.linear_d2 = linear_softplus(128, 64) # 64, softplus
        self.linear_d3 = linear_softplus(64, 64) # 64, softplus
        self.linear_d4 = linear_softplus(64, 128) # 128, SAAF

        ## deconv
        self.deconv = nn.ConvTranspose1d(128, 1, kernel_size=63, padding=(63-1)//2)
        self.out_act = nn.Tanh()

    def forward(self, x, is_bypass=False):
        # adptive front-end
        # print('x >>', x.size())
        x1 = self.conv1_e1(x)
        # print('x1 >>', x1.size())
        x1_abs = torch.abs(x1)

        x2 = self.conv2_e2(x1_abs)
        x2 = self.conv_act(x2)
        # print('x2 >>', x2.size())
        z, pool_idx = self.maxpool(x2)

        # bottleneck
        # print('z >>', z.size())

        z = self.linear_z1(z)
        z_hat = self.linear_z2(z)
        # print('z_hat >>', z_hat.size())

        # Ssnthesis back-end
        x2_hat = self.unpool(z_hat, pool_idx)
        # print('x2_hat >>', x2_hat.size())
        x1_hat = x2_hat * x1

        # DNN-SAAF
        x1_hat = x1_hat.permute(0, 2, 1)

        tensor = self.linear_d1(x1_hat)
        tensor = self.linear_d2(tensor)
        tensor = self.linear_d3(tensor)
        tensor= self.linear_d4(tensor)
        x0_hat = self.linear_d4(tensor)

        x0_hat = x0_hat.permute(0, 2, 1)

        # deconv
        y_hat = self.deconv(x0_hat)
        return self.out_act(y_hat)


    def compute_loss(self, anno, pred):
        return F.l1_loss(anno, pred)

    def run_on_batch(self, batch):
        wav_x = batch['x']
        # print('model: ', wav_x.size()) # (32, 1, 1024)
        wav_y = batch['y']
        # print('model: ', wav_y.size()) # (32, 1, 1024)

        wav_y_pred = self(wav_x)

        wav_y = torch.squeeze(wav_y)
        wav_y_pred = torch.squeeze(wav_y_pred)
        loss = self.compute_loss(wav_y, wav_y_pred)
        return wav_y_pred, loss


if __name__ == '__main__':
    batch_size = 32
    win_len = 1024
    tensor_in = torch.zeros(batch_size, 1, win_len)
    print('tnesor_in:', tensor_in.size())

    model = Net()
    tensor_out = model(tensor_in)
    print('tnesor_out:', tensor_out.size())
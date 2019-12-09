import torch

SR = 16000
WIN_LEN = 1024
HOP_LEN = 64

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
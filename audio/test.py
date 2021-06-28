#!/usr/bin/env python
# coding=utf-8

# *************************************************************************
#	> File Name: test.py
#	> Author: Yang Zhang 
#	> Mail: zyziszy@foxmail.com
#	> Created Time: Mon Mar 29 16:21:06 2021
# ************************************************************************/


import numpy as np
import torch
from torch.nn import functional as F
from scipy.io import wavfile
from scipy import signal

def gaussian_filter(sig, kernel_size=5, std=1):
    kernel = signal.windows.gaussian(kernel_size, std)
    kernel = torch.FloatTensor(kernel/float(kernel_size)).unsqueeze(0).unsqueeze(0)
    sig = sig.unsqueeze(0).unsqueeze(0)
    sig = F.conv1d(sig, kernel, padding=2)
    return sig.reshape(1, -1)

for i in [1, 2, 3]:
    src_path = "adv_target_{}.wav".format(i)
    dst_path = "adv_voting_60_target_{}.wav".format(i)

    sample_rate, audio = wavfile.read(src_path)
    audio = torch.FloatTensor(audio)
    gaussian_nosie = torch.randn_like(audio)*120
    snr = 10*torch.log10( torch.mean(audio**2) / torch.mean(gaussian_nosie**2) )
    audio = audio + gaussian_nosie
    audio = audio.cpu().detach().numpy()
    #wavfile.write(dst_path, sample_rate, audio.astype(np.int16))

for i in [1, 2, 3]:
    src_path = "adv_nontarget_{}.wav".format(i)
    dst_path = "adv_voting_60_nontarget_{}.wav".format(i)

    sample_rate, audio = wavfile.read(src_path)
    audio = torch.FloatTensor(audio)
    gaussian_nosie = torch.randn_like(audio) * 60
    audio = audio + gaussian_nosie
    audio = audio.cpu().detach().numpy()
    #wavfile.write(dst_path, sample_rate, audio.astype(np.int16))

for i in [1, 2, 3]:
    src_path = "adv_target_{}.wav".format(i)
    dst_path = "adv_filter_60_target_{}.wav".format(i)

    sample_rate, audio = wavfile.read(src_path)
    audio = torch.FloatTensor(audio)
    audio = gaussian_filter(audio)[0]
    audio = audio.cpu().detach().numpy()
    #wavfile.write(dst_path, sample_rate, audio.astype(np.int16))

for i in [1, 2, 3]:
    src_path = "adv_nontarget_{}.wav".format(i)
    dst_path = "adv_filter_60_nontarget_{}.wav".format(i)

    sample_rate, audio = wavfile.read(src_path)
    audio = torch.FloatTensor(audio)
    audio = gaussian_filter(audio)[0]
    audio = audio.cpu().detach().numpy()
    #wavfile.write(dst_path, sample_rate, audio.astype(np.int16))


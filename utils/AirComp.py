#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import numpy as np
import torch
from scipy import integrate
import scipy.stats as stats

def ei_calculation(gth):
    invexp = lambda x: 1/x * np.exp(-x)
    y, error = integrate.quad(invexp, gth, np.inf)
    return y

def AirComp_4QAM_AWGN(g_temp,snr):
    x_input = g_temp.cpu().numpy()
    Quan = (x_input > 0.).astype(float) + (x_input < 0.).astype(float)*(-1.)
    noise = np.random.normal(0.0, 1.0, size=len(Quan))
    x_output = np.sqrt((10.0 ** (snr / 10.0)))*Quan/np.sqrt(2.0) + noise/np.sqrt(2.)
    y_temp = torch.from_numpy(x_output)
    y = y_temp.cuda().float()
    return y

# Fading with Perfect CSI
def AirComp_4QAM_Fading(g_temp,snr,g_th,Ei):
    x_input = g_temp.cpu().numpy()
    Dim = x_input.size

    e0 = np.e
    p = 1 - e0 ** (-1 * g_th)
    zeta = np.random.binomial(1, p, size=None)
    if zeta == 1:
        x_output = np.zeros([Dim, 1])
    else:
        Quan = (x_input > 0.).astype(float) + (x_input < 0.).astype(float) * (-1.)
        noise = np.random.normal(0.0, 1.0, size=len(Quan))
        x_output = np.sqrt((10.0 ** (snr / 10.0)) / Ei) * Quan / np.sqrt(2.0) + noise / np.sqrt(2.)

    y_temp = torch.from_numpy(x_output)
    y = y_temp.cuda().float()
    return y


def AirComp_4QAM_Imperfect_CSI(g_temp,snr,thd,Ei,g_th):
    x_input = g_temp.cpu().numpy()
    Dim = x_input.size
    mu = 0
    sigma = 1
    # g_th = np.pi
    e0 = np.e
    p = 1 - e0 ** (-1 * g_th)
    zeta = np.random.binomial(1, p, size=None)
    if zeta == 1:
        x_output = np.zeros([Dim, 1])
    else:
        Quan = (x_input > 0.).astype(float) + (x_input < 0.).astype(float) * (-1.)
        x = 1 / np.sqrt(2.0) * Quan[::2] + 1j * Quan[1:2]
        noise_Re = np.random.normal(0.0, 1.0, size=len(Quan)//2)
        noise_Im = np.random.normal(0.0, 1.0, size=len(Quan)//2)
        noise = 1 / np.sqrt(2.0) * noise_Re + 1j * noise_Im
        h_Re = np.random.normal(0.0, 1.0, size=len(Quan)//2)
        h_Im = np.random.normal(0.0, 1.0, size=len(Quan)//2)
        h = 1 / np.sqrt(2.0) * h_Re + 1j * h_Im
        lower, upper = mu - np.sqrt(thd/2.0), mu + np.sqrt(thd/2.0)
        X = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
        Delta_Re = X.rvs(len(Quan)//2)
        Delta_Im = X.rvs(len(Quan)//2)
        Delta = Delta_Re + 1j * Delta_Im
        h_est = h + Delta
        y = np.sqrt((10.0 ** (snr / 10.0)) / Ei) * 1 / h_est * h * x + noise
        y_Re = y.real
        y_Im = y.imag

        # import pdb
        # pdb.set_trace()

        y_cat = np.concatenate((np.expand_dims(y_Re, 1), np.expand_dims(y_Im, 1)), axis=1)
        x_output = np.squeeze(np.reshape(y_cat, (1, Dim)), 0)




    y_temp = torch.from_numpy(x_output)
    y = y_temp.cuda().float()
    return y









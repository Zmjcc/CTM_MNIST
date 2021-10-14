#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch

from .AirComp import AirComp_4QAM_AWGN, AirComp_4QAM_Imperfect_CSI, AirComp_4QAM_Fading, ei_calculation


def quan_average_gradients_AWGN(w,snr):
    
    w_avg = copy.deepcopy(w[0])
    index_to_size = []
    index_to_key = []
    cumsum = [0]
    for key in w_avg.keys():
        w_avg[key].zero_()
        index_to_size.append(w_avg[key].size())
        index_to_key.append(key)
        cumsum.append(cumsum[-1] + w_avg[key].numel())
    #import ipdb; ipdb.set_trace()
    for i in range(0, len(w)):
        temp_flat = []
        for k in w_avg.keys():
            temp_flat.append(w[i][k].view(-1))

        temp_flat = torch.cat(temp_flat, -1)
        temp_flat_quan = AirComp_4QAM_AWGN(temp_flat,snr)

        for j in range(len(cumsum)-1):
            begin = cumsum[j]
            end = cumsum[j+1]
            temp_flat_raw = temp_flat_quan[begin:end].view(*index_to_size[j])
            w[i][index_to_key[j]] = temp_flat_raw

    for k in w_avg.keys():
        for i in range(0, len(w)):
            w_avg[k] += w[i][k]

        # w_avg[k] = torch.div(w_avg[k], len(w))
        mask_neg_all = w_avg[k] < 0.
        mask_pos_all = w_avg[k] > 0.
        quan_all = mask_neg_all.float() * (-1.0) + mask_pos_all.float() * 1.0
        w_avg[k].zero_().add_(quan_all)
    import ipdb; ipdb.set_trace()
    return w_avg

def quan_average_gradients_Fading(w,snr,g_th):
    w_avg = copy.deepcopy(w[0])
    index_to_size = []
    index_to_key = []
    cumsum = [0]
    for key in w_avg.keys():
        w_avg[key].zero_()
        index_to_size.append(w_avg[key].size())
        index_to_key.append(key)
        cumsum.append(cumsum[-1] + w_avg[key].numel())

    for i in range(0, len(w)):
        temp_flat = []
        for k in w_avg.keys():
            temp_flat.append(w[i][k].view(-1))

        temp_flat = torch.cat(temp_flat, -1)

        # Nu = ei_calculation(np.pi)
        # De = ei_calculation(g_th)
        Ei = ei_calculation(g_th)
        temp_flat_quan = AirComp_4QAM_Fading(temp_flat,snr,g_th,Ei)
        # temp_flat_quan = AirComp_4QAM_Fading_Analog(temp_flat, snr, g_th)

        for j in range(len(cumsum)-1):
            begin = cumsum[j]
            end = cumsum[j+1]
            temp_flat_raw = temp_flat_quan[begin:end].view(*index_to_size[j])
            w[i][index_to_key[j]] = temp_flat_raw

    for k in w_avg.keys():
        for i in range(0, len(w)):
            w_avg[k] += w[i][k]

        # w_avg[k] = torch.div(w_avg[k], len(w))
        mask_neg_all = w_avg[k] < 0.
        mask_pos_all = w_avg[k] > 0.
        quan_all = mask_neg_all.float() * (-1.0) + mask_pos_all.float() * 1.0
        w_avg[k].zero_().add_(quan_all)

    return w_avg

def quan_average_gradients_Imperfect_CSI(w,snr,delta,g_th):
    w_avg = copy.deepcopy(w[0])
    index_to_size = []
    index_to_key = []
    cumsum = [0]
    for key in w_avg.keys():
        w_avg[key].zero_()
        index_to_size.append(w_avg[key].size())
        index_to_key.append(key)
        cumsum.append(cumsum[-1] + w_avg[key].numel())

    for i in range(0, len(w)):
        temp_flat = []
        for k in w_avg.keys():
            temp_flat.append(w[i][k].view(-1))
        temp_flat = torch.cat(temp_flat, -1)

        Ei = ei_calculation(g_th)
        temp_flat_quan = AirComp_4QAM_Imperfect_CSI(temp_flat,snr,delta,Ei,g_th)

        for j in range(len(cumsum)-1):
            begin = cumsum[j]
            end = cumsum[j+1]
            temp_flat_raw = temp_flat_quan[begin:end].view(*index_to_size[j])
            w[i][index_to_key[j]] = temp_flat_raw

    for k in w_avg.keys():
        for i in range(0, len(w)):
            w_avg[k] += w[i][k]

        # w_avg[k] = torch.div(w_avg[k], len(w))
        mask_neg_all = w_avg[k] < 0.
        mask_pos_all = w_avg[k] > 0.
        quan_all = mask_neg_all.float() * (-1.0) + mask_pos_all.float() * 1.0
        w_avg[k].zero_().add_(quan_all)

    return w_avg

def average_gradients_schedule(w,dataset_ratio_list,scheduling_probabiliy_list):
    #import ipdb; ipdb.set_trace()
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        w_avg[key].zero_()
    for k in w_avg.keys():
        for i in range(0, len(w)):
            #import ipdb; ipdb.set_trace()
            w_avg[k] += w[i][k]*dataset_ratio_list[i]/scheduling_probabiliy_list[i]
    return w_avg
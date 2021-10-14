#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse
import numpy as np

def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--epochs', type=int, default=300, help="rounds of training")
    parser.add_argument('--num_users', type=int, default=100, help="number of users: K")
    parser.add_argument('--frac', type=float, default= 1.0, help="the fraction of clients: C")
    parser.add_argument('--local_ep', type=int, default=1, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=512, help="local batch size: B")
    parser.add_argument('--bs', type=int, default=512, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0, help="SGD momentum (default: 0.5)")
    parser.add_argument('--split', type=str, default='user', help="train-test split type, user or sample")

    # model arguments
    parser.add_argument('--model', type=str, default='cnn', help='model name')
    parser.add_argument('--kernel_num', type=int, default=9, help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to use for convolution')
    parser.add_argument('--norm', type=str, default='batch_norm', help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32, help="number of filters for conv nets")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than strided convolutions")

    # other arguments
    parser.add_argument('--dataset', type=str, default='mnist', help="name of dataset")
    parser.add_argument('--iid', action='store_true', help='whether i.i.d or not')
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--num_channels', type=int, default=3, help="number of channels of imges")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--stopping_rounds', type=int, default=10, help='rounds of early stopping')
    parser.add_argument('--verbose', action='store_true', help='verbose print')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--lr_scheduler', action='store_true', help='whether using lr scheduler')

    # quan
    parser.add_argument('--mode', type=str, default='AWGN', help="Tx mode")
    # snr
    parser.add_argument('--snr', type=int, default=-10, help="SNR")
    # delta: channel estimation error
    parser.add_argument('--delta', type=float, default= 0.01, help="Delta")
    # g_th
    parser.add_argument('--thd', type=float, default= 4.2, help="THD")


    #schedule policy
    parser.add_argument('--schedule_policy', type=str, default= 'based_on_dataset_size', help="schedule policy")
    parser.add_argument('--schedule_user_num', type=int, default=1, help="user number to upload the gradient")
    parser.add_argument('--T_LC_T_BC', type=float, default=0, help="user number to upload the gradient")
    parser.add_argument('--nu', type=int, default=10, help="user number to upload the gradient")
    parser.add_argument('--rho', type=float, default=5e-3, help="hyperparameter of Rjk's paper")
    parser.add_argument('--differ_label',type = str , default=  '',help = "used to differ same schedule policy")
    args = parser.parse_args()
    return args

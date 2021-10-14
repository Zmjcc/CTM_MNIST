#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import torch
import math
from torch import nn
from torch.utils.data import DataLoader, Dataset
from collections import OrderedDict
from utils.util import calculate_gradient_norm
from copy import deepcopy
from models.test import test_img
class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None):

        self.args = args
        if args.model !='svm':
            self.loss_func = nn.CrossEntropyLoss(reduction='sum')
        else:
            from models.Nets import SVM_Loss
            self.loss_func = SVM_Loss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)

    def train(self, net):
        net.train()

        epoch_loss = []
        net.zero_grad()

        for iter in range(self.args.local_ep):
            batch_loss = []
            sample_count = 0
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)

                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                sample_count += len(images)

                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/sample_count)

        grad_dict = OrderedDict()
        for (key, value) in net.named_parameters():
            grad_dict[key] = value.grad.data.detach()/(self.args.local_ep*sample_count)

        buffer_dict = OrderedDict()
        for (key, module) in net.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                buffer_dict[key]=OrderedDict()
                buffer_dict[key]['running_mean'] = module._buffers['running_mean'].data.detach()
                buffer_dict[key]['running_var'] = module._buffers['running_var'].data.detach()
                buffer_dict[key]['num_batches_tracked'] = module._buffers['num_batches_tracked'].data.detach()

        return grad_dict, buffer_dict, sum(epoch_loss) / len(epoch_loss)
    def train_est_lipsthiz(self,net,lr,args,dataset_train):
        net.train()

        epoch_loss = []
        net.zero_grad()
        optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=args.momentum)
        for iter in range(self.args.local_ep):
            batch_loss = []
            sample_count = 0
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)

                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                sample_count += len(images)

                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/sample_count)

        grad_dict = OrderedDict()
        for (key, value) in net.named_parameters():
            grad_dict[key] = value.grad.data.detach()/(self.args.local_ep*sample_count)
        buffer_dict = OrderedDict()
        for (key, module) in net.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                buffer_dict[key]=OrderedDict()
                buffer_dict[key]['running_mean'] = module._buffers['running_mean'].data.detach()
                buffer_dict[key]['running_var'] = module._buffers['running_var'].data.detach()
                buffer_dict[key]['num_batches_tracked'] = module._buffers['num_batches_tracked'].data.detach()

        for (key, value) in net.named_parameters():
            value.grad = value.grad/(self.args.local_ep*sample_count) 
        optimizer.step()
        net.eval()
        updated_epoch_loss = []
        optimizer.zero_grad()

        for iter in range(self.args.local_ep):
            batch_loss = []
            sample_count = 0
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)

                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                sample_count += len(images)
                batch_loss.append(loss.item())
            updated_epoch_loss.append(sum(batch_loss)/sample_count)
        l_est = 2*((updated_epoch_loss[-1]-epoch_loss[-1])+lr*calculate_gradient_norm([grad_dict]))/(lr**2*(calculate_gradient_norm([grad_dict])))
        return grad_dict, buffer_dict, sum(epoch_loss) / len(epoch_loss), l_est



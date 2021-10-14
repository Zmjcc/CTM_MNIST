import copy
import numpy as np
from torchvision import datasets, transforms
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.sampling import random_pick_user, mnist_noniid_given_dataset_ratio, cifar_noniid,random_pick_multiuser
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar, CNNCifarRes, CNNCifarRes18, ALLCNN
from utils.averaging import average_gradients_schedule

from utils.util import calculate_transmite_time, user_channel_state_information
from utils.schedule_policy import Random_scheduling,CA_scheduling,IA_scheduling,ICA_scheduling,CTM_scheduling

from models.test import test_img
import scipy.io as io
from copy import deepcopy
import time
import logging

import os

random_argument_root = './random_argument.mat'
# 创建一个logger
logger = logging.getLogger('mytest')
logger.setLevel(logging.DEBUG)

args = args_parser()

# 创建一个handler，用于写入日志文件
file_name = '%s_schedule_num_%d%s'%(str(args.schedule_policy),args.schedule_user_num,args.differ_label)
fh = logging.FileHandler('./log/' + file_name +'.log')
fh.setLevel(logging.DEBUG)

# 再创建一个handler，用于输出到控制台
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

# 定义handler的输出格式
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)

# 给logger添加handler
logger.addHandler(fh)
logger.addHandler(ch)

logger.info('fed_learn_mnist_cnn_non_iid')

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    # parse args
    args.device = torch.device(0)
    total_communication_time = 0

    train_loss_list = []
    test_accuracy_list = []
    communication_time_list = []
    acc_list = []
    if random_argument_root !='':
        #dataset_size_ratio = io.loadmat(random_argument_root)['dataset_size_ratio'][0,:]
        T_E = io.loadmat(random_argument_root)['T_E']
        channel_gain_list = io.loadmat(random_argument_root)['channel_gain_list']

    dataset_size_ratio = np.ones(args.num_users)/args.num_users
    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
        # make the user's dataset same in different ways
        try:
            dict_users = np.load('dict_users.npy',allow_pickle=True)
        except:
            dict_users, _ = mnist_noniid_given_dataset_ratio(dataset_train, args.num_users, dataset_size_ratio)
            np.save('dict_users.npy', dict_users)
        dict_users = dict_users.item()

    else:
        exit('Error: unrecognized dataset')
    img_size = dataset_train[0][0].shape

    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'resnet18' and args.dataset == 'cifar':
        net_glob = CNNCifarRes18(args=args).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = ALLCNN(input_size=3).to(args.device)
    else:
        exit('Error: unrecognized model')
    print(net_glob)
    net_glob.train()

    w_glob = net_glob.state_dict()

    # training
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []

    # 记录日志
    logger.info(args)
    optimizer = torch.optim.SGD(net_glob.parameters(), lr=args.lr, momentum=args.momentum)
    if args.lr_scheduler:
        lambda1 = lambda epoch: 1 / (epoch + args.nu)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda1)

    net_total_params = sum(p.numel() for p in net_glob.parameters())
    print('| net_total_params:', net_total_params)

    if args.dataset == 'mnist':
        log_probs_dummy = net_glob(torch.ones(1, 1, 28, 28).to(args.device))
    else:
        log_probs_dummy = net_glob(torch.ones(1, 3, 32, 32).to(args.device))
    loss_dummy = F.cross_entropy(log_probs_dummy, torch.ones(1, ).cuda().long())
    loss_dummy.backward()
    optimizer.zero_grad()

    # 生成信道参数
    user_CSI = user_channel_state_information(user_num=args.num_users,
                                              distance=np.random.uniform(0.3, 0.7, args.num_users), bandwidth=10 ** 6,
                                              transmite_power=10 ** (24 / 10), channel_variance=1)
    expect_transmite_time = T_E

    for iter in range(1, args.epochs+1):
        w_locals, loss_locals = [], []
        buffer_locals = []
        l_est_locals = []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        for idx in idxs_users:
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            if args.schedule_policy == 'CTM':
                w, buffer, loss, l_est = local.train_est_lipsthiz(net=copy.deepcopy(net_glob).to(args.device),
                                                                  lr=optimizer.state_dict()['param_groups'][0]['lr'],
                                                                  args=args, dataset_train=dataset_train)
                l_est_locals.append(l_est)
            else:
                w, buffer, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
            w_locals.append(copy.deepcopy(w))

            loss_locals.append(copy.deepcopy(loss))
            buffer_locals.append(copy.deepcopy(buffer))

        this_round_channel_gain = channel_gain_list[:,(iter-1)%1000]
        if args.schedule_policy =='random':
            scheduling_probability_list = Random_scheduling(dataset_size_ratio)
        elif args.schedule_policy =='CA':
            scheduling_probability_list = CA_scheduling(transmite_power=user_CSI.transmite_power,
                                                                          noise_power=user_CSI.noise_power,
                                                                          bandwidth=user_CSI.bandwidth,
                                                                          channel_gain=this_round_channel_gain)
        elif args.schedule_policy =='IA':
            scheduling_probability_list = IA_scheduling(dataset_size_ratio,w_locals)
        elif args.schedule_policy =='ICA':
            scheduling_probability_list = ICA_scheduling(
                dataset_size_ratio=dataset_size_ratio, w_locals=w_locals, transmite_power=user_CSI.transmite_power,
                noise_power=user_CSI.noise_power, bandwidth=user_CSI.bandwidth, channel_gain=this_round_channel_gain,
                rho=args.rho, q=16, S=net_total_params)
        elif args.schedule_policy == 'CTM':
            #estimate l
            l_est_this_round = max(l_est_locals)[0]
            if l_est_this_round < 0:
                l_est_this_round = 3.24
            logger.info("l_est_this_round: {:.2f}".format(l_est_this_round))
            scheduling_probability_list = CTM_scheduling(
                dataset_size_ratio=dataset_size_ratio, w_locals=w_locals, transmite_power=user_CSI.transmite_power,
                noise_power=user_CSI.noise_power, bandwidth=user_CSI.bandwidth, channel_gain=this_round_channel_gain,
                q=16, S=net_total_params, lipschitz_constant=l_est_this_round,
                learning_rate=optimizer.state_dict()['param_groups'][0]['lr'], nu=args.nu,
                expect_transmite_time=expect_transmite_time + args.T_LC_T_BC, convergence_epsilon=0.5,
                round_index_now=iter)
        else:
            raise NotImplementedError

        user_index = random_pick_multiuser(scheduling_probability_list,args.schedule_user_num)
        w_locals_choosen = []
        for i in range(len(user_index)):
            w_locals_choosen.append(w_locals[user_index[i]])

        w_glob = average_gradients_schedule(w_locals_choosen, dataset_size_ratio[user_index],
                                            scheduling_probability_list[user_index])
        total_communication_time += max(calculate_transmite_time(transmite_power=user_CSI.transmite_power[user_index],
                                                                 noise_power=user_CSI.noise_power[user_index],
                                                                 bandwidth=user_CSI.bandwidth[user_index],
                                                                 channel_gain=this_round_channel_gain[user_index], q=16,
                                                                 S=net_total_params))

        for key, value in net_glob.named_parameters():
            value.grad.data = w_glob[key].data.detach()


        def average_buffer(w, layer):
            w_avg = copy.deepcopy(w[0][layer])
            for k in w_avg.keys():
                for i in range(1, len(w)):
                    w_avg[k] += w[i][layer][k]
                w_avg[k] = torch.div(w_avg[k], len(w))
            return w_avg


        for (key, module) in net_glob.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                buffer_avg = average_buffer(buffer_locals, key)
                module._buffers['running_mean'].data = buffer_avg['running_mean'].data
                module._buffers['running_var'].data = buffer_avg['running_var'].data
                module._buffers['num_batches_tracked'].data = buffer_avg['num_batches_tracked'].data

        optimizer.step()

        if args.lr_scheduler:
            scheduler.step()

        if iter == 1:
            train_loss_list.append(2.3)
        else:
            train_loss_list.append(loss_avg)

        loss_avg = sum(loss_locals) / len(loss_locals)

        logger.info('Epoch: {}'.format(iter))
        logger.info('Train loss: {:.4f}'.format(loss_avg))
        logger.info('learning_rate:{:.4f}'.format(optimizer.state_dict()['param_groups'][0]['lr']))

        del w_locals, loss_locals, buffer_locals
        # testing

        acc_test, loss_test = test_img(net_glob, dataset_test, args)
        logger.info('Test accuracy: {:.4f}'.format(acc_test))
        communication_time_list.append(deepcopy(total_communication_time))
        acc_list.append(acc_test)
        logger.info("average communication time: {:.4f}".format(np.max(total_communication_time)))

    io.savemat('./log/' + file_name + '.mat',
               {'train_loss': np.array(train_loss_list), 'test_accuracy': np.array(acc_list),
                'communication_time': np.array(communication_time_list)})
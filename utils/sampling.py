#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import numpy as np
from torchvision import datasets, transforms
from copy import deepcopy
#np.random.seed(2020)
def mnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    #import ipdb; ipdb.set_trace()
    return dict_users


def mnist_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    #import ipdb
    #ipdb.set_trace()
    num_shards, num_imgs = 200, 300
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users


def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def cifar_noniid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """

    #import ipdb;ipdb.set_trace()
    num_items = 330
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    labels = np.array(dataset.targets)
    idxs = np.arange(len(dataset))
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs_class_0 = idxs_labels[0,:5000]
    idxs_class_1 = idxs_labels[0,5000:10000]

    for i in range(num_users//2):
        class_0_random_choose = np.random.choice(idxs_class_0, num_items, replace=False)
        class_1_random_choose = np.random.choice(idxs_class_1, num_items, replace=False)
        dict_users[i*2 ] = set(np.concatenate([class_0_random_choose]))
        dict_users[i*2+1] = set(np.concatenate([class_1_random_choose]))
        idxs_class_0 = list(set(idxs_class_0) - set(class_0_random_choose))
        idxs_class_1 = list(set(idxs_class_1) - set(class_1_random_choose))
    return dict_users




def mnist_iid_user_different(dataset, num_users,local_batch_size):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    #num_items = int(len(dataset)/num_users)
    total_batch = int(len(dataset)/local_batch_size)
    user_batch_num = np.zeros(num_users,dtype = np.int)
    user_random_num = (np.random.rand(num_users))
    user_random_num = user_random_num/np.sum(user_random_num)
    for user in range(num_users):
        user_batch_num[user] = int(total_batch/num_users/4) + int(3/4*total_batch*user_random_num[user])
    user_batch_num[np.random.randint(0,num_users)] += total_batch - np.sum(user_batch_num)
    #import ipdb; ipdb.set_trace()
    user_data_num = user_batch_num * local_batch_size
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, user_data_num[i], replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    #import ipdb; ipdb.set_trace()
    return dict_users,user_batch_num/total_batch



def mnist_noniid_user_different(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    #import ipdb
    #ipdb.set_trace()
    num_shards, num_imgs = 300, 200
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels.numpy()

    user_shards_num = np.zeros(num_users,dtype = np.int)
    user_random_shards_num = (np.random.rand(num_users))
    user_random_shards_num = user_random_shards_num/np.sum(user_random_shards_num)
    for user in range(num_users):
        user_shards_num[user] = int(num_shards/num_users/2) + int(1/2*num_shards*user_random_shards_num[user])
    now_user_shards_num = np.sum(user_shards_num)
    for lost in range(num_shards-now_user_shards_num):
        user_shards_num[np.random.randint(0,num_users)] +=1
    # sort labels

    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, user_shards_num[i], replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users,user_shards_num/num_shards

def mnist_noniid_given_dataset_ratio(dataset, num_users,dataset_size_ratio):
    num_shards, num_imgs = 300, 200
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels.numpy()
    user_shards_num = dataset_size_ratio * num_shards
    user_shards_num = user_shards_num.astype(np.int32)

    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, user_shards_num[i], replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users,user_shards_num/num_shards





def random_pick_user(probabilities):
    x=np.random.uniform(0,1)
    cumulative_probability=0.0
    index = range(len(probabilities))
    for item,item_probability in zip(index,probabilities):
        cumulative_probability+=item_probability
        if x < cumulative_probability: break
    return item


def random_pick_multiuser(probabilities,schedule_number):
    picked_user_index_list = []
    probability_list = deepcopy(probabilities)
    for i in range(schedule_number):
        picked_user_index = random_pick_user(probabilities = probability_list)
        picked_user_index_list.append(picked_user_index)
        # set the picked user probability to zeros to avoid being picked twice
        probability_list[picked_user_index] = 0
        probability_list = probability_list/np.sum(probability_list)
    return picked_user_index_list





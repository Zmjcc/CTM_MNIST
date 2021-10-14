import numpy as np
import torch
import copy
def calculate_gradient_norm(w):
    user_num = len(w)
    gradient_norm = np.zeros(user_num)
    for i in range(user_num):
        temp_flat = []
        for k in w[0].keys():
            temp_flat.append(w[i][k].view(-1))
        temp_flat = torch.cat(temp_flat, -1)
        gradient_norm[i] = torch.norm(temp_flat,2)**2
    return gradient_norm
def calculate_transmite_time(transmite_power,noise_power,bandwidth,channel_gain,q,S):
    user_num = len(channel_gain)
    transmite_time_list = np.zeros(user_num)
    for user in range(user_num):
        transmite_time_list[user] = q*S/(bandwidth[user]*np.log2(1+transmite_power[user]*channel_gain[user]/noise_power[user]))
    return transmite_time_list

class user_channel_state_information():
    def __init__(self, user_num,distance,bandwidth,transmite_power,channel_variance):
        self.user_num = user_num
        self.noise_power = np.ones(user_num) * bandwidth * 10**(-174/10)
        self.path_loss = 10**(-(128.1+37.6*np.log10(distance))/10)
        self.channel_variance = 1
        self.transmite_power = np.ones(user_num)*transmite_power
        self.bandwidth = np.ones(user_num)*bandwidth
    def generate_one_round_channel_gain(self):
        return abs(np.random.randn(self.user_num) + 1j*np.random.randn(self.user_num))**2/2*self.channel_variance *self.path_loss
    def get_expect_transmite_time(self,dataset_size_ratio,q,S):
        expect_transmite_time = 0
        for iteration_round in range(100000):
            expect_transmite_time += np.sum(dataset_size_ratio*calculate_transmite_time(self.transmite_power,self.noise_power,self.bandwidth,self.generate_one_round_channel_gain(),q,S))
        return expect_transmite_time/100000
                
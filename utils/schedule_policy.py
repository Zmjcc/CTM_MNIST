import numpy as np
from utils.util import calculate_gradient_norm,calculate_transmite_time
def Random_scheduling(dataset_size_ratio):
    return dataset_size_ratio
def CA_scheduling(transmite_power,noise_power,bandwidth,channel_gain,schedule_num=1):
    user_num = len(channel_gain)
    transmite_time_one_gradient_param_list = np.zeros(user_num)
    schedule_probability_list = np.zeros(user_num)
    for user in range(user_num):
        transmite_time_one_gradient_param_list[user] = 1/(bandwidth[user]*np.log2(1+transmite_power[user]*channel_gain[user]/noise_power[user]))
    user_permit_transmite = np.argsort(transmite_time_one_gradient_param_list)[:schedule_num].tolist()
    schedule_probability_list[user_permit_transmite] = np.ones(schedule_num)/schedule_num
    return schedule_probability_list
def IA_scheduling(dataset_size_ratio,w_locals):
    gradient_norm_list = dataset_size_ratio * np.sqrt(calculate_gradient_norm(w_locals))
    schedule_probability_list = gradient_norm_list/sum(gradient_norm_list)
    return schedule_probability_list
def ICA_scheduling(dataset_size_ratio,w_locals,transmite_power,noise_power,bandwidth,channel_gain,rho,q,S):

    def calculate_probability_list(dataset_size_ratio,gradient_norm_list,transmite_time_list,rho,lagrangian_factor):
        user_num = len(transmite_time_list)
        schedule_probability_list = np.zeros(user_num)
        for user in range(user_num):
            schedule_probability_list[user] = dataset_size_ratio[user] * np.sqrt(gradient_norm_list[user]) * np.sqrt(rho/((1-rho)*transmite_time_list[user] + lagrangian_factor))
        return schedule_probability_list
    user_num = len(channel_gain)
    gradient_norm_list = calculate_gradient_norm(w_locals)
    transmite_time_list = calculate_transmite_time(transmite_power,noise_power,bandwidth,channel_gain,q,S)
    schedule_probability_list = np.zeros(user_num)
    # bi searching for lambda
    interation_round = 0
    lower_bound = -min((1-rho)*transmite_time_list)
    upper_bound = 1
    while np.sum(calculate_probability_list(dataset_size_ratio,gradient_norm_list,transmite_time_list,rho,lagrangian_factor=upper_bound)) >= 1:
        upper_bound *=2
    lagrangian_factor = (lower_bound + upper_bound)/2
    probability_sum_now = np.sum(calculate_probability_list(dataset_size_ratio,gradient_norm_list,transmite_time_list,rho,lagrangian_factor))
    while abs(probability_sum_now-1) >= 0.01:
        if probability_sum_now>1:
            lower_bound = lagrangian_factor
        else:
            upper_bound = lagrangian_factor
        lagrangian_factor = (lower_bound + upper_bound)/2
        probability_sum_now = np.sum(calculate_probability_list(dataset_size_ratio,gradient_norm_list,transmite_time_list,rho,lagrangian_factor))
        interation_round +=1
        if interation_round>5000:
            break
    schedule_probability_list = calculate_probability_list(dataset_size_ratio,gradient_norm_list,transmite_time_list,rho,lagrangian_factor)
    return schedule_probability_list/np.sum(schedule_probability_list)
def CTM_scheduling(dataset_size_ratio,w_locals,transmite_power,noise_power,bandwidth,channel_gain,q,S,lipschitz_constant,learning_rate,nu,expect_transmite_time,convergence_epsilon,round_index_now):
    def calculate_probability_list(dataset_size_ratio,gradient_norm_list,transmite_time_list,learning_rate,nu,lipschitz_constant,expect_transmite_time,convergence_epsilon,lagrangian_factor,round_index_now):
        user_num = len(transmite_time_list)
        schedule_probability_list = np.zeros(user_num)
        for user in range(user_num):
            schedule_probability_list[user] = dataset_size_ratio[user] * np.sqrt(lipschitz_constant*(round_index_now+nu+1)*learning_rate**2*gradient_norm_list[user]*expect_transmite_time/(2*convergence_epsilon*(transmite_time_list[user]+lagrangian_factor)))
        return schedule_probability_list
    user_num = len(channel_gain)
    gradient_norm_list = calculate_gradient_norm(w_locals)
    transmite_time_list = calculate_transmite_time(transmite_power,noise_power,bandwidth,channel_gain,q,S)
    schedule_probability_list = np.zeros(user_num)
    lower_bound = -min(transmite_time_list)
    upper_bound = 1
    interation_round = 0
    while np.sum(calculate_probability_list(dataset_size_ratio,gradient_norm_list,transmite_time_list,learning_rate,nu,lipschitz_constant,expect_transmite_time,convergence_epsilon,lagrangian_factor = upper_bound,round_index_now = round_index_now )) >= 1:
        upper_bound *=2
    lagrangian_factor = (lower_bound + upper_bound)/2
    probability_sum_now = np.sum(calculate_probability_list(dataset_size_ratio,gradient_norm_list,transmite_time_list,learning_rate,nu,lipschitz_constant,expect_transmite_time,convergence_epsilon,lagrangian_factor,round_index_now))
    while abs(probability_sum_now-1) >= 0.0001:
        if probability_sum_now>1:
            lower_bound = lagrangian_factor
        else:
            upper_bound = lagrangian_factor
        lagrangian_factor = (lower_bound + upper_bound)/2
        probability_sum_now = np.sum(calculate_probability_list(dataset_size_ratio,gradient_norm_list,transmite_time_list,learning_rate,nu,lipschitz_constant,expect_transmite_time,convergence_epsilon,lagrangian_factor,round_index_now))
        interation_round +=1
    schedule_probability_list = calculate_probability_list(dataset_size_ratio,gradient_norm_list,transmite_time_list,learning_rate,nu,lipschitz_constant,expect_transmite_time,convergence_epsilon,lagrangian_factor,round_index_now)
    return schedule_probability_list/np.sum(schedule_probability_list)
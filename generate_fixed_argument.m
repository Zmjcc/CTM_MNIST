data_length = 1000;
user_num = 30;
transimit_power = 24;%dBm
noise_power = 10^6 * 10^(-174/10);
%generate user distance per user
user_distance = 0.3 + (0.7-0.3)*rand(1,user_num);
%generate dataset size ratio per user
total_shards = 300;
user_shards_list = zeros(1,user_num);
user_random_shards_num = rand(1,user_num); user_random_shards_num = user_random_shards_num/sum(user_random_shards_num);
for user=1:user_num
    user_shards_list(user) = 2*total_shards/user_num/5 + floor(3*total_shards/5* user_random_shards_num(user));
end
for shard_list = 1:(total_shards-sum(user_shards_list))
    random_choose_a_user = unidrnd(user_num);
    user_shards_list(random_choose_a_user) = user_shards_list(random_choose_a_user)+1;
end
dataset_size_ratio = user_shards_list/total_shards;

%generate channel gain per user
PL = 10.^(-(128.1 + 37.6.*log10(user_distance))/10);
channel_gain_list = zeros(user_num,data_length);
real_R_reciprocal_list = zeros(user_num,data_length);
expected_R_reciprocal_list = zeros(1,user_num);
for user=1:user_num
    h_beixuan = sqrt(PL(user))*(1/sqrt(2)*normrnd(0,1,[1,data_length*2])+1j*1/sqrt(2)*normrnd(0,1,[1,data_length*2]));
    h_2_beixuan = abs(h_beixuan).^2;
    truncted_point = PL(user)/1000;
    h_2 = h_2_beixuan(h_2_beixuan>=truncted_point);
    h_2 = h_2(1:data_length);
    channel_gain_list(user,:) = h_2;
    SNR = 10^(transimit_power/10)./noise_power;
    real_R_reciprocal_list(user,:) = 1./(log2(1+SNR.*h_2));
    %ti = 16*583240./(10^6.*(log2(1+SNR.*h_2)));
    %ti = 1./(log2(1+SNR.*h_2));
    fun = @(x,PL,SNR) (exp(-x./PL)./(PL.*log2(1+SNR.*x)));
    expected_R_reciprocal_list(user) = integral(@(x) fun(x,PL(user),SNR),truncted_point,PL(user)*15)/(expcdf(PL(user)*15,PL(user))-expcdf(truncted_point,PL(user)));
    %mean_time = mean(ti); 
end
T_E = sum(16*582026*dataset_size_ratio.*expected_R_reciprocal_list/10^6);
save('random_argument.mat','channel_gain_list','user_distance','dataset_size_ratio','expected_R_reciprocal_list','T_E');
%calculate expected transimited time
%sum(16*582026*dataset_size_ratio.*expected_R_reciprocal_list/10^6)

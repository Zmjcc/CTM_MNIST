#Communication-efficient federated edge learning via optimal probabilistic device scheduling
This is the source code for implementing the paper of “Communication-efficient federated edge learning via optimal probabilistic device scheduling”, 
in this paper, we proposed an optimized scheduling policy for total communication time minimization.

One experiments in the paper is included.

1) CNN on MNIST dataset

#Requirements
python = 3.6.5 

pytorch

#Commands for running the experiments:
```python main_fed_multiuser.py --dataset mnist  --num_channels 1 --model cnn --num_users 30 --epochs 1000 --lr 1 --nu 100 --frac 1.0 --local_ep 1 --momentum 0.0 --local_bs 50 --bs 50 --mode 'schedule' --schedule_policy 'CTM' --schedule_user_num 1 --lr_schedule --differ_label '_v10' --gpu 1 --rho 0.5```

#Note
1) The output of the experiment is a .log file including all the training results, e.g. training loss and test accuracy; 
   
2) The configurable parameters in the code are defined in the file `option.py`.
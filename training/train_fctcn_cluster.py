import sys
sys.path.insert(0, '../')
import torch
import mocap.datasets.h36m as H36M
import numpy as np
from torch.utils.data import DataLoader
from numpy import random
from time import time
from mocap.mlutil.sequence import PoseDataset
from fcgan.trainer.fctcn_trainer import SegmentForecasting_Trainer
from mocap.datasets.custom_activities import CustomActivities

force_new_training = True
model_seed = 0
device = torch.device("cuda")

# PARAMS
# ================================
train_batchsize = 64
test_batchsize = 1024
# --
n_in = 12
n_out = 20
hidden_units = 64
framerate = 12.5
txt = 'h36m_cluster8'
if framerate != 25:
    txt += '_fr' + str(framerate)
label_dim = 8
stacks = 3
dim = 96


ds_train = H36M.H36M_FixedSkeleton(
        remove_global_Rt=True,
        actors=['S1', 'S6', 'S7', 'S8', 'S9', 'S11'])
ds_test = H36M.H36M_FixedSkeleton(
        remove_global_Rt=True,
        actors=['S5'])
ds_train = CustomActivities(ds_train, n_activities=label_dim,
            activity_dir='/home/user/visualize/h36m_cluster' + str(label_dim), 
            postfix='_cluster' + str(label_dim))
ds_test = CustomActivities(ds_test, n_activities=label_dim,
            activity_dir='/home/user/visualize/h36m_cluster' + str(label_dim), 
            postfix='_cluster' + str(label_dim))


ds_train = PoseDataset(
        ds_train, n_frames=n_in+n_out, framerates=[framerate],
        add_noise=True, noise_var=0.001, mirror_data=True)
ds_test = PoseDataset(
        ds_test, n_frames=n_in+n_out, framerates=[framerate],
        add_noise=False, mirror_data=False)

print()
print('#train samples', len(ds_train))
print('#val samples', len(ds_test))

dl_train = DataLoader(ds_train, batch_size=train_batchsize,
                      shuffle=True, num_workers=8)

dl_test = DataLoader(ds_test, batch_size=test_batchsize,
                     shuffle=False, num_workers=8)

print()
print("#train batches", len(dl_train))
print("#test batches", len(dl_test))

# TRAINER
# ================================
trainer = SegmentForecasting_Trainer(
    n_in=n_in, n_out=n_out,
    hidden_units=hidden_units, device=device, label_dim=label_dim, stacks=stacks,
    txt=txt, model_seed=model_seed, force_new_training=force_new_training, dim=dim
)
print("#params:", trainer.prettyprint_number_of_parameters())

optim1 = torch.optim.Adam(trainer.models[0].parameters(), lr=0.0005, amsgrad=True, weight_decay=0)
scheduler1 = torch.optim.lr_scheduler.ExponentialLR(optim1, gamma=.92)

optim2 = torch.optim.Adam(trainer.models[1].parameters(), lr=0.0005, amsgrad=True, weight_decay=0)
scheduler2 = torch.optim.lr_scheduler.ExponentialLR(optim2, gamma=.96)

trainer.run(dl_train, dl_test, optim=[optim1, optim2], optim_scheduler=[scheduler1, scheduler2])

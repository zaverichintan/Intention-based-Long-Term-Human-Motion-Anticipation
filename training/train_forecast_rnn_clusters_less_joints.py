import sys
sys.path.insert(0, '../')
import torch
import mocap.datasets.h36m as H36M
import numpy as np
from torch.utils.data import DataLoader
from numpy import random
from time import time
from mocap.mlutil.sequence import PoseDataset
from fcgan.trainer.forecast_rnn_trainer import ForecastRNN_Trainer
from mocap.datasets.custom_activities import CustomActivities
from os.path import join, isdir, isfile, abspath
import pathlib

force_new_training = True
model_seed = 0
device = torch.device("cuda")

# best: 1 stack, 64 dims:
# loss:5.6827 accX:0.8700 accY:0.8387

# PARAMS
# ================================
train_batchsize = 256
test_batchsize = 1024
# --
txt = '_cluster_less_joints'
n_in = 25
n_out = 50
hidden_units = 12
hidden_units_disc = 32
framerate = 12.5
stacks = 3
label_dim = 8
dim = 51

# activity_dir = '/home/user/CVPR21_forecastgan/data/h36m_clusters8'
local_path = abspath(pathlib.Path(__file__).parent.absolute())
activity_dir = abspath(join(local_path, '../data/clustered_labels'))+'/h36m/h36m_clusters8'

ds_train = H36M.H36M_FixedSkeletonFromRotation(
        remove_global_Rt=True,
        actors=['S1', 'S6', 'S7', 'S8', 'S9', 'S11'])

ds_test = H36M.H36M_FixedSkeletonFromRotation(
        remove_global_Rt=True,
        actors=['S5'])

ds_train = H36M.H36M_Simplified(ds_train)
ds_test = H36M.H36M_Simplified(ds_test)

ds_train = CustomActivities(ds_train, activity_dir=activity_dir, n_activities=8)
ds_test = CustomActivities(ds_test, activity_dir=activity_dir, n_activities=8)

ds_train = PoseDataset(ds_train, n_frames=n_in+n_out, framerates=[framerate],
        add_noise=True, noise_var=0.001, mirror_data=True)

ds_test = PoseDataset(ds_test, n_frames=n_in+n_out, framerates=[framerate],
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
trainer = ForecastRNN_Trainer(
    n_in=n_in, n_out=n_out, hidden_units=hidden_units, 
    device=device, label_dim=label_dim, txt=txt, 
    model_seed=model_seed, stacks=stacks,
    hidden_units_disc=hidden_units_disc,
    force_new_training=force_new_training, dim=dim
)
print("#params:", trainer.prettyprint_number_of_parameters())

params1 = trainer.models[0].parameters()
optim1 = torch.optim.Adam(params1, lr=0.0005, amsgrad=True, weight_decay=0)
scheduler1 = torch.optim.lr_scheduler.ExponentialLR(optim1, gamma=.92)

params2 = trainer.models[1].parameters()
optim2 = torch.optim.Adam(params2, lr=0.0005, amsgrad=True, weight_decay=0)
scheduler2 = torch.optim.lr_scheduler.ExponentialLR(optim2, gamma=.92)


trainer.run(dl_train, dl_test, optim=[optim1, optim2], 
            optim_scheduler=[scheduler1, scheduler2])

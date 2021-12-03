import sys
sys.path.insert(0, '../')
import torch
import mocap.datasets.h36m as H36M
from mocap.datasets.combined import Combined
import numpy as np
from torch.utils.data import DataLoader
from numpy import random
from time import time
from mocap.mlutil.sequence import PoseDataset
from fcgan.trainer.ae_clustering_trainer import ClusteringTrainer

force_new_training = True
model_seed = 0
device = torch.device("cuda")

# PARAMS
# ================================
train_batchsize = 32
test_batchsize = 1024
# --
hidden_units = 128
txt = 'h36m_activity11'
label_dim = 11
dim = 42

# DATA
# ================================
ds_train = Combined(H36M.H36M_FixedSkeleton(
        remove_global_Rt=True,
        actors=['S1', 'S6', 'S7', 'S8', 'S9', 'S11']))
ds_test = Combined(H36M.H36M_FixedSkeleton(
        remove_global_Rt=True,
        actors=['S5']))

ds_train = PoseDataset(
        ds_train, n_frames=3, framerates=[12.5],
        add_noise=True, noise_var=0.001, mirror_data=True)
ds_test = PoseDataset(
        ds_test, n_frames=3, framerates=[12.5],
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
trainer = ClusteringTrainer(
    hidden_units=hidden_units, dim=dim, model_seed=model_seed,
    device=device, txt=txt, force_new_training=force_new_training
)
print("#params:", trainer.prettyprint_number_of_parameters())

optim = torch.optim.Adam(trainer.models[0].parameters(), lr=0.0005, amsgrad=True, weight_decay=0)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=.96)

trainer.run(dl_train, dl_test, optim=optim, optim_scheduler=scheduler)



import sys

sys.path.insert(0, '../')
import torch
import mocap.datasets.h36m as H36M
import numpy as np
from torch.utils.data import DataLoader
from numpy import random
from time import time
from mocap.mlutil.sequence import PoseDataset
from ourgan.trainer.ourgan import Ourgan_trainer
from mocap.datasets.custom_activities import CustomActivities
from ourgan.data_loader.data_loader import load_train_test_data
from ourgan.config import config

force_new_training = True

device = torch.device("cuda")

# PARAMS
# ================================

activity_dir = config.activity_dir + 'h36m/h36m_naiveclusters' + str(config.label_dim)

dl_train, dl_test = load_train_test_data()
print()
print("#train batches", len(dl_train))
print("#test batches", len(dl_test))

# TRAINER
# ================================
trainer = Ourgan_trainer(
	n_in=config.n_in, n_out=config.n_out,
	hidden_units=config.hidden_units, device=config.device, label_dim=config.label_dim, stacks=config.stacks,
	txt=config.experiment_name, model_seed=config.model_seed, force_new_training=config.force_new_training, dim=config.data_dim
)
print("#params:", trainer.prettyprint_number_of_parameters())

# encoder
optim1 = torch.optim.Adam(trainer.models[0].parameters(), lr=0.0001, amsgrad=True, weight_decay=0)
scheduler1 = torch.optim.lr_scheduler.ExponentialLR(optim1, gamma=.92)
# decoder
optim2 = torch.optim.Adam(trainer.models[1].parameters(), lr=0.0001, amsgrad=True, weight_decay=0)
scheduler2 = torch.optim.lr_scheduler.ExponentialLR(optim2, gamma=.96)
# discriminator
optim3 = torch.optim.Adam(trainer.models[2].parameters(), lr=0.0001, amsgrad=True, weight_decay=0)
scheduler3 = torch.optim.lr_scheduler.ExponentialLR(optim3, gamma=.96)

trainer.run(dl_train, dl_test, optim=[optim1, optim2, optim3], optim_scheduler=[scheduler1, scheduler2, scheduler3])

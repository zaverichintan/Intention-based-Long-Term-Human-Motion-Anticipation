import sys
sys.path.insert(0, '../')
import torch
import mocap.datasets.h36m as H36M
import mocap.evaluation.h36m as H36M_EV
import numpy as np
from torch.utils.data import DataLoader
from numpy import random
from time import time
from os.path import join, isdir
from os import makedirs
import shutil
from tqdm.auto import tqdm
from mocap.mlutil.sequence import PoseDataset
# from fcgan.trainer.forecast_trainer import Forecast_Trainer
from fcgan.forecasting import forecaster

model_seed = 0
device = torch.device("cuda")

# PARAMS
# ================================
# train_batchsize = 64
# test_batchsize = 1024
# # --
txt = ''
n_in = 12
n_out = 12
# hidden_units = 64
# framerate = 12.5
# stacks = 1
# label_dim = 8
# dim = 96

# TRAINER
# ================================
# trainer = Forecast_Trainer(
#     n_in=n_in, n_out=n_out, hidden_units=hidden_units, 
#     device=device, label_dim=label_dim, txt=txt, 
#     model_seed=model_seed, stacks=stacks,
#     force_new_training=False, dim=dim
# )
# E = trainer.models[0]
# E.load_specific_weights('weights_ep0000.h5')
# D = trainer.models[1]
# D.load_specific_weights('weights_ep0000.h5')

# print("#params:", trainer.prettyprint_number_of_parameters())

vis_root = '/home/user/visualize/forecast' + txt
if isdir(vis_root):
    shutil.rmtree(vis_root)
makedirs(vis_root)

for action in tqdm(H36M.ACTIONS):
    Seq, Labels = H36M_EV.get(action, H36M.H36M_FixedSkeleton_withSimplifiedActivities)
    Seq = Seq[:, :50]
    Labels = Labels[:, 50:]
    n_out = Labels.shape[1]
    
    # Labels_pred = trainer.predict(Seq, n_out)
    Labels_pred = forecaster.forecast_simplified_activities(Seq, n_out=n_out)

    np.save(join(vis_root, action + '_pred.npy'), Labels_pred)
    np.save(join(vis_root, action + '_gt.npy'), Labels)

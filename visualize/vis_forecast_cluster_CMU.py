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
from fcgan.trainer.forecast_trainer import Forecast_Trainer
from mocap.datasets.custom_activities import CustomActivities
from ourgan.forecasting import forecaster_CMU
from mocap.datasets.cmu_eval import CMUEval, CMUEval3D, DataType
from ourgan.config import config
from ourgan.data_loader_vis import load_vis_data

model_seed = 0
device = torch.device("cuda")

# PARAMS
# ================================
ACTIVITES = ['basketball',
'basketball_signal',
'directing_traffic',
'jumping',
# 'running',
'soccer',
# 'walking',
'walking_extra',
'washwindow']

vis_root = '/media/data/zaveri/CVPR21/visualize_labels/forecast_cluster_CMU'
if config.dataset_type == 'Euler':
    vis_root = vis_root+'_Euler'

if isdir(vis_root):
    shutil.rmtree(vis_root)
makedirs(vis_root)
config.num_seeds = 256
for action in tqdm(ACTIVITES):
    Seqs, Labels = load_vis_data(action, num_seeds=config.num_seeds)
    Seqs = Seqs[:, :50]
    Labels = Labels[:, :50]
    n_out = Labels.shape[1]


    if config.dataset_type == 'Euler':
        Labels_pred = forecaster_CMU.forecast_clusters_exp(Seqs, n_out=n_out)
    elif config.dataset_type == '3D':
        Labels_pred = forecaster_CMU.forecast_clusters(Seqs, n_out=n_out)

    np.save(join(vis_root, action + '_pred.npy'), Labels_pred)
    np.save(join(vis_root, action + '_gt.npy'), Labels)

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
from fcgan.forecasting import forecaster

model_seed = 0
device = torch.device("cuda")

# PARAMS
# ================================


vis_root = '/home/user/visualize/forecast_cluster'
if isdir(vis_root):
    shutil.rmtree(vis_root)
makedirs(vis_root)

activity_dir = '/home/user/CVPR21_forecastgan/data/h36m_clusters8'

for action in tqdm(H36M.ACTIONS):
    Seq, Labels = H36M_EV.get(
        action, H36M.H36M_FixedSkeleton, 
        Wrapper_fn=lambda ds: CustomActivities(ds, activity_dir=activity_dir, n_activities=8))
    Seq = Seq[:, :50]
    Labels = Labels[:, :50]
    n_out = Labels.shape[1]
    
    # Labels_pred = trainer.predict(Seq, n_out)
    Labels_pred = forecaster.forecast_clusters(Seq, n_out=n_out)

    np.save(join(vis_root, action + '_pred.npy'), Labels_pred)
    np.save(join(vis_root, action + '_gt.npy'), Labels)

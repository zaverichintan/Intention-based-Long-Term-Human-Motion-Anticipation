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
from ourgan.trainer.ourgan import Ourgan_trainer
from ourgan.config import config
from ourgan.data_loader_vis import load_vis_data
from fcgan.forecasting import forecaster
from ourgan.forecasting import forecaster_CMU
from mocap.visualization.sequence import SequenceVisualizer
from utils.forward_kinematics import forward_kinematics, forward_kinematicsCMU, forward_kinematics_expmap_only
from mocap.datasets.cmu_eval import CMUEval, CMUEval3D, ACTIVITES, DataType
import mocap.datasets.amass_constants.test as AMASS_TEST
import mocap.datasets.amass_constants.train as AMASS_TRAIN
from mocap.datasets.amass import AMASS

model_seed = 0
device = torch.device("cuda")

n_in = 10
n_out = 1200

train = AMASS_TRAIN.FILES

test = AMASS_TEST.FILES

train_leftover_datasets = [
"BMLmovi",
"EKUT",
"KIT",
"MPI_mosh",
"TCD_handMocap",
"SFU",
"TotalCapture",
"BMLhandball",
"DFaust_67",
]

# data_loc = '/media/zaveri/amass2skel'
data_loc = '/home/zaveri_cvg21/Data/amass/amass2skel'

train_leftover_datasets = []

ds = AMASS(data_loc, datasets=[], exact_files=test)

print(config.experiment_name)

trainer = Ourgan_trainer(
	n_in=config.n_in, n_out=config.n_out,
	hidden_units=config.hidden_units, device=config.device, label_dim=config.label_dim, stacks=config.stacks,
	txt=config.experiment_name, model_seed=config.model_seed, force_new_training=False, dim=config.data_dim
)

for j in range(len(ds)):
    seq = ds[j]
    Seq = []
    for t in range(0, len(seq)-config.n_in, 20):
        Seq.append(seq[t:t+config.n_in])

    Seq = np.array(Seq)
    n_batch = len(Seq)
    print(n_batch)
    auto_regress_vis = 60

    n_output_lables =  config.n_out*(auto_regress_vis+1) + config.forecast

    Y_labels = np.zeros((n_batch, n_output_lables, 8), dtype=np.float32)
    noise = torch.randn(1, n_batch, config.noise_dim, device=config.device) * config.noise_factor
    
    Y_hat_old = trainer.predict(Seq, Y_labels[:, config.n_in:config.n_frames + config.forecast, :], noise)
    Y_hatcom = Y_hat_old

    for i in range(60):
        start = config.n_out * i
        start_forecast = (config.n_out - config.n_in) * (2 * i + 1)
        end = start_forecast + config.n_in
        Y_hat = trainer.predict(Y_hat_old[:, config.n_out - config.n_in:, :],
                                Y_labels[:, start:end + config.forecast, :],
                                noise)
        Y_hatcom = torch.cat((Y_hatcom, Y_hat), 1)
        Y_hat_old = Y_hat
    Y_hat = Y_hatcom
    print(Y_hat.shape)

    # vis_dir = "/media/remote_home/zaveri/Downloads/amass_output/video/"
    # if not isdir(vis_dir):
    #     makedirs(vis_dir)
    # vis = SequenceVisualizer(vis_dir, "vis_amass_new", to_file=True)

    # output_seq = Y_hat[0].cpu().detach().numpy()
    # print(output_seq.shape)
    # vis.plot(output_seq, create_video=True, plot_jid=False)

    np.save("/media/remote_home/zaveri/Downloads/amass_output/files/amass_test_%04d.npy" % j, Y_hat.cpu().detach().numpy())
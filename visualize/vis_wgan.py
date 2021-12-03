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
# from ourgan.trainer.ourgan_without_disc import Ourgan_trainer # for wo disc
from ourgan.config import config
from ourgan.data_loader_vis import load_vis_data
from fcgan.forecasting import forecaster
from ourgan.forecasting import forecaster_CMU
from mocap.visualization.sequence import SequenceVisualizer
from utils.forward_kinematics import forward_kinematics, forward_kinematicsCMU
from mocap.datasets.cmu_eval import CMUEval, CMUEval3D, ACTIVITES, DataType

model_seed = 0
device = torch.device("cuda")

# PARAMS
# ================================
txt = config.experiment_name
if config.long_term_inference:
	auto_regress_vis = config.long_term_auto_regress_vis
else:
	auto_regress_vis = config.auto_regress_vis
wgan_training = True
if wgan_training:
	config.experiment_name = join('wgan', config.experiment_name)
	config.epoch_to_visualize = str(70)

# TRAINER
# ================================
trainer = Ourgan_trainer(
	n_in=config.n_in, n_out=config.n_out,
	hidden_units=config.hidden_units, device=config.device, label_dim=config.label_dim, stacks=config.stacks,
	txt=config.experiment_name, model_seed=config.model_seed, force_new_training=False, dim=config.data_dim
)

E = trainer.models[0]
E.load_weights_for_epoch(int(config.epoch_to_visualize))
D = trainer.models[1]
D.load_weights_for_epoch(int(config.epoch_to_visualize))

print("#params:", trainer.prettyprint_number_of_parameters())

vis_root = config.vis_root
file_root = config.file_root
if isdir(vis_root):
	shutil.rmtree(vis_root)
makedirs(vis_root)
if isdir(file_root):
	shutil.rmtree(file_root)
makedirs(file_root)

if config.dataset_type == 'Euler':
	file_root_euler = join(file_root, join('Euler'))
	if isdir(file_root_euler):
		shutil.rmtree(file_root_euler)
	makedirs(file_root_euler)
	file_root_3D = join(file_root, join('3D'))
	if isdir(file_root_3D):
		shutil.rmtree(file_root_3D)
	makedirs(file_root_3D)


OUT_EUC = {}
OUT_MSE = {}
OUT_EUC_A = []
OUT_MSE_A = []
if config.dataset_name == 'H36M':
	ACTIONS = H36M.ACTIONS
	ITEMS = np.array([1, 3, 7, 9, 13, 15, 17, 24, 49, 74, 99])

elif config.dataset_name == 'CMU':
	ACTIONS = ['basketball', 'basketball_signal', 'directing_traffic',
    'jumping', 'soccer', 'walking', 'walking_extra',
    'washwindow']
	ITEMS = np.array([1, 3, 7, 9, 13, 15, 17, 24, 49, 74, 99])

if config.long_term_inference:
	auto_regress_vis = config.long_term_auto_regress_vis
else:
	auto_regress_vis = config.auto_regress_vis

for action in ACTIONS:
	Seqs, Labels = load_vis_data(action, num_seeds=config.num_seeds)
	Seq_for_labels = Seqs[:, :50, :]
	n_output_lables = (config.auto_regress_vis + 1) * config.n_out

	if config.labels_type == 'simplified' and config.dataset_name == 'H36M' and config.dataset_type == '3D':
		Y_labels = forecaster.forecast_simplified_activities(Seq_for_labels, n_out=n_output_lables)
	elif config.labels_type == 'clustered' and config.dataset_name == 'H36M' and config.dataset_type == '3D':
		Y_labels = forecaster.forecast_clusters(Seq_for_labels, n_out=n_output_lables)
	elif config.labels_type == 'simplified' and config.dataset_name == 'H36M' and config.dataset_type == 'Euler':
		Y_labels = forecaster.forecast_simplified_activities_exp(Seq_for_labels, n_out=n_output_lables)
	elif config.labels_type == 'clustered' and config.dataset_name == 'H36M' and config.dataset_type == 'Euler':
		Y_labels = forecaster.forecast_clusters_exp(Seq_for_labels, n_out=n_output_lables)

	elif config.labels_type == 'clustered' and config.dataset_name == 'CMU' and config.dataset_type == '3D':
		Y_labels = forecaster_CMU.forecast_clusters(Seq_for_labels, n_out=n_output_lables)
	elif config.labels_type == 'clustered' and config.dataset_name == 'CMU' and config.dataset_type == 'Euler':
		Y_labels = forecaster_CMU.forecast_clusters_exp(Seq_for_labels, n_out=n_output_lables)


	X = Seqs[:, :config.n_in, :]
	Y = Seqs[:, config.n_in:, :]
	print(X.shape)
	print(Y.shape)

	if (config.labels_type == 'zero'):
		Y_labels = Y_labels * 0

	noise = torch.randn(1, config.num_seeds, config.noise_dim, device=config.device)
	Y_hat_old = trainer.predict(X, Y_labels[:, config.n_in:config.n_frames + config.forecast, :], noise)
	Y_hatcom = Y_hat_old

	for i in range(auto_regress_vis):
		start = config.n_out * i
		start_forecast = (config.n_out - config.n_in) * (2 * i + 1)
		end = start_forecast + config.n_in
		Y_hat = trainer.predict(Y_hat_old[:, config.n_out - config.n_in:, :],
		                        Y_labels[:, start:end + config.forecast, :],
		                        noise)
		Y_hatcom = torch.cat((Y_hatcom, Y_hat), 1)
		Y_hat_old = Y_hat
	Y_hat = Y_hatcom

	# convert to 3D coordinates
	if config.dataset_name == 'H36M' and config.dataset_type == 'Euler':
		X, Y, Y_hat, X_Euler, Y_Euler, Y_hat_Euler = forward_kinematics(X, Y, Y_hat.cpu())
	elif config.dataset_name == 'CMU' and config.dataset_type == 'Euler':
		X, Y, Y_hat, X_Euler, Y_Euler, Y_hat_Euler = forward_kinematicsCMU(X, Y, Y_hat.cpu())
	elif config.dataset_type == '3D':
		Y_hat = Y_hat.cpu().numpy()

	if config.plot_videos:
		vis = SequenceVisualizer(vis_root, 'vis_posegen_' + action, to_file=True)
		indices = np.arange(start=1, stop=7, step=1)
		for i in indices:
			x = X[i]
			y = Y[i]
			y_hat = Y_hat[i]
			vis.plot(seq1=x, seq2=y_hat, name='prediction', video_fps=12.5, create_video=True)
			vis.plot(seq1=y, seq2=y_hat, name='gt', video_fps=12.5, create_video=True)

	if config.dataset_type == 'Euler':
		fname_input = join(file_root_euler, action + '_gt.npy')
		fname_pred = join(file_root_euler, action + '_pred.npy')
		inp_euler = np.concatenate((X_Euler, Y_Euler), axis=1)
		np.save(fname_input, inp_euler)
		np.save(fname_pred, Y_hat_Euler)

		fname_input = join(file_root_3D, action + '_gt.npy')
		fname_pred = join(file_root_3D, action + '_pred.npy')
		inp = np.concatenate((X, Y), axis=1)
		np.save(fname_input, inp)
		np.save(fname_pred, Y_hat)

	elif config.dataset_type == '3D':
		fname_input = join(file_root, action + '_gt.npy')
		fname_pred = join(file_root, action + '_pred.npy')
		inp = np.concatenate((X, Y), axis=1)
		np.save(fname_input, inp)
		np.save(fname_pred, Y_hat)

	distance = H36M_EV.calculate_euclidean_distance(Y, Y_hat)
	if config.dataset_type == 'Euler':
		sq_sum = np.sqrt(np.sum((Y_hat_Euler[:, :, 3:] - Y_Euler[:, :, 3:]) ** 2, axis=2))
		mse = np.mean(np.array(sq_sum), axis=0)
		OUT_MSE[action] = mse
		OUT_MSE_A.append(mse[ITEMS])
	OUT_EUC[action] = distance
	OUT_EUC_A.append(distance[ITEMS])

# Print the results
print('  80   160   320   400   560   640   720   1000   2000   3000   4000')
for action, metric in OUT_EUC.items():
	print(action)
	print(np.round(metric[ITEMS] * 1000, 1))

if config.dataset_type == 'Euler':
	for action, metric in OUT_MSE.items():
		print(action)
		print(np.round(metric[ITEMS], 3))

print('\n Average Joints error ')
OUT_EUC_A_np = np.array(OUT_EUC_A)
print('  80   160   320   400   560   640   720   1000   2000   3000   4000')
print(np.round(np.mean(OUT_EUC_A_np, axis=0) * 1000, 1))

if config.dataset_type == 'Euler':
	print('\n Average Angular error ')
	OUT_MSE_A_np = np.array(OUT_MSE_A)
	print('  80   160   320   400   560   640   720   1000   2000   3000   4000')
	print(np.round(np.mean(OUT_MSE_A_np, axis=0), 3))

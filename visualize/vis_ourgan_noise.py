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

model_seed = 0
device = torch.device("cuda")

# PARAMS
# ================================
txt = config.experiment_name
if config.long_term_inference:
	auto_regress_vis = config.long_term_auto_regress_vis
else:
	auto_regress_vis = config.auto_regress_vis
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

vis_root = join(config.vis_root, join(str(config.number_of_noises), str(config.noise_factor)))
file_root = join(config.file_root, join(str(config.number_of_noises), str(config.noise_factor)))

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

ITEMS = np.array([1, 3, 7, 9, 13, 15, 17, 24, 49, 74, 99])
OUT_EUC = {}
OUT_MSE = {}
OUT_EUC_A = []
OUT_MSE_A = []
if config.dataset_name == 'H36M':
	ACTIONS = H36M.ACTIONS
	# ACTIONS = ['discussion', 'eating', 'walking', 'smoking']
if config.dataset_name == 'H36M_less_joints':
	ACTIONS = H36M.ACTIONS
if config.dataset_name == 'H36M_less_joints_fixed':
	ACTIONS = H36M.ACTIONS

elif config.dataset_name == 'CMU':
	ACTIONS = ['basketball', 'basketball_signal', 'directing_traffic',
	           'jumping', 'soccer', 'walking', 'walking_extra',
	           'washwindow']
			   
if config.long_term_inference:
	auto_regress_vis = config.long_term_auto_regress_vis
else:
	auto_regress_vis = config.auto_regress_vis

for action in ACTIONS:
	X_all_noise = []
	Y_all_noise = []
	Y_hat_all_noise = []
	X_Euler_all_noise = []
	Y_Euler_all_noise = []
	Y_Euler_hat_all_noise = []

	mse_error_full_noises = []
	distance_all_noise = []
	mse_all_noise = []


	Seqs, Labels = load_vis_data(action, num_seeds=config.num_seeds)
	Seq_for_labels = Seqs[:, :50, :]
	n_output_lables =  config.n_out*(auto_regress_vis+1) + config.forecast

	if config.labels_type == 'simplified' and config.dataset_name == 'H36M' and config.dataset_type == '3D':
		Y_labels = forecaster.forecast_simplified_activities(Seq_for_labels, n_out=n_output_lables)
	elif config.labels_type == 'clustered' and config.dataset_name == 'H36M' and config.dataset_type == '3D':
		Y_labels = forecaster.forecast_clusters(Seq_for_labels, n_out=n_output_lables)
	elif config.labels_type == 'simplified' and config.dataset_name == 'H36M' and config.dataset_type == 'Euler':
		Y_labels = forecaster.forecast_simplified_activities_exp(Seq_for_labels, n_out=n_output_lables)
	elif config.labels_type == 'clustered' and config.dataset_name == 'H36M' and config.dataset_type == 'Euler' and config.label_dim == 8:
		Y_labels = forecaster.forecast_clusters_exp(Seq_for_labels, n_out=n_output_lables)
	elif config.labels_type == 'naive_clustered' and config.dataset_name == 'H36M' and config.dataset_type == 'Euler':
		Y_labels = forecaster.forecast_naive_clusters_exp(Seq_for_labels, n_out=n_output_lables)
	elif config.labels_type == 'zero' and config.dataset_name == 'H36M' and config.dataset_type == 'Euler':
		Y_labels = forecaster.forecast_simplified_activities_exp(Seq_for_labels, n_out=n_output_lables)

	elif config.labels_type == 'zero' and config.dataset_name == 'H36M' and config.dataset_type == '3D':
		Y_labels = forecaster.forecast_simplified_activities(Seq_for_labels, n_out=n_output_lables)

	elif config.labels_type == 'clustered' and config.dataset_name == 'H36M_less_joints' and config.dataset_type == '3D':
		Y_labels = forecaster.forecast_clusters_less_joints(Seq_for_labels, n_out=n_output_lables)
		# print(Y_labels.shape) 
		# Y_labels = np.concatenate((Y_labels[:,:10,:], Labels), axis=1)
	elif config.labels_type == 'clustered' and config.dataset_name == 'H36M_less_joints_fixed' and config.dataset_type == '3D':
		Y_labels = forecaster.forecast_clusters_less_joints(Seq_for_labels, n_out=n_output_lables)
		
	elif config.labels_type == 'zero' and config.dataset_name == 'CMU' and config.dataset_type == 'Euler':
		Y_labels = forecaster_CMU.forecast_clusters_exp(Seq_for_labels, n_out=n_output_lables)


	elif config.labels_type == 'clustered' and config.dataset_name == 'CMU' and config.dataset_type == '3D':
		Y_labels = forecaster_CMU.forecast_clusters(Seq_for_labels, n_out=n_output_lables)
	elif config.labels_type == 'clustered' and config.dataset_name == 'CMU' and config.dataset_type == 'Euler':
		Y_labels = forecaster_CMU.forecast_clusters_exp(Seq_for_labels, n_out=n_output_lables)


	if config.labels_type_ablation == 'nogan' and config.dataset_name == 'H36M' and config.dataset_type == 'Euler':
		Y_labels = forecaster.forecast_clusters_exp_nogan(Seq_for_labels, n_out=n_output_lables)
	elif config.labels_type_ablation == 'onlygan' and config.dataset_name == 'H36M' and config.dataset_type == 'Euler':
		Y_labels = forecaster.forecast_clusters_exp_onlygan(Seq_for_labels, n_out=n_output_lables)

	# Ablation of 4 and 12 labels
	if config.labels_type == 'clustered' and config.dataset_name == 'H36M' and config.dataset_type == 'Euler' and config.label_dim == 4:
		Y_labels = forecaster.forecast_clusters_exp4(Seq_for_labels, n_out=n_output_lables)
	elif config.labels_type == 'clustered' and config.dataset_name == 'H36M' and config.dataset_type == 'Euler' and config.label_dim == 12:
		Y_labels = forecaster.forecast_clusters_exp12(Seq_for_labels, n_out=n_output_lables)


	X_in = Seqs[:, :config.n_in, :]
	Y_in = Seqs[:, config.n_in:, :]
	
	for noise_number in range(config.number_of_noises):
		if (config.labels_type == 'zero'):
			Y_labels = Y_labels * 0

		noise = torch.randn(1, config.num_seeds, config.noise_dim, device=config.device) * config.noise_factor
		Y_hat_old = trainer.predict(X_in, Y_labels[:, config.n_in:config.n_frames + config.forecast, :], noise)
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
			# X, Y, Y_hat, X_Euler, Y_Euler, Y_hat_Euler = forward_kinematics(X_in, Y_in, Y_hat.cpu())
			X, Y, Y_hat, X_Euler, Y_Euler, Y_hat_Euler = forward_kinematics_expmap_only(X_in, Y_in, Y_hat.cpu())
		elif config.dataset_name == 'CMU' and config.dataset_type == 'Euler':
			X, Y, Y_hat, X_Euler, Y_Euler, Y_hat_Euler = forward_kinematicsCMU(X_in, Y_in, Y_hat.cpu())
		elif config.dataset_type == '3D':
			Y_hat = Y_hat.cpu().numpy()

		if config.plot_videos:
			vis = SequenceVisualizer(vis_root, 'vis_posegen_' + action, to_file=True)
			indices = np.arange(start=0, stop=8, step=1)
			for i in indices:
				x = X_in[i]
				y = Y_in[i]
				y = y[:config.frames_to_visualize]
				y_hat = Y_hat[i]
				y_hat = y_hat[:config.frames_to_visualize]

				vis.plot(seq1=x, seq2=y_hat, name=action + '_' + str(noise_number) + '_' + str(i), video_fps=25, create_video=True)


		# add the seqs to all noise seqs
		if config.dataset_type == 'Euler':
			X_all_noise.append(X_in)
			Y_all_noise.append(Y_in)
			Y_hat_all_noise.append(Y_hat)

			X_Euler_all_noise.append(X_Euler)
			Y_Euler_all_noise.append(Y_Euler)
			Y_Euler_hat_all_noise.append(Y_hat_Euler)
	#
	# 		distance = H36M_EV.calculate_euclidean_distance(Y, Y_hat)
	# 		distance_all_noise.append(distance)
	# 		sq_sum = np.sqrt(np.sum((Y_hat_Euler[:, :, 3:] - Y_Euler[:, :, 3:]) ** 2, axis=2))
	# 		mse = np.mean(np.array(sq_sum), axis=0)
	# 		mse_all_noise.append(mse)
	#
		if config.dataset_type == '3D':
			X_all_noise.append(X_in)
			Y_all_noise.append(Y_in)
			Y_hat_all_noise.append(Y_hat)
			# distance = H36M_EV.calculate_euclidean_distance(Y_in, Y_hat)
			# distance_all_noise.append(distance)

	
	X_all_noise = np.array(X_all_noise)
	Y_all_noise = np.array(Y_all_noise)
	Y_hat_all_noise = np.array(Y_hat_all_noise)

	X_Euler_all_noise = np.array(X_Euler_all_noise)
	Y_Euler_all_noise = np.array(Y_Euler_all_noise)
	Y_Euler_hat_all_noise = np.array(Y_Euler_hat_all_noise)
	#
	# distance_all_noise = np.array(distance_all_noise)
	# distance_all_noise_mean = np.mean(distance_all_noise, axis=0)
	# distance_all_noise_mean = np.min(distance_all_noise, axis=0)
	# OUT_EUC[action] = distance_all_noise_mean
	# OUT_EUC_A.append(distance_all_noise_mean[ITEMS])
	# if config.dataset_type == 'Euler':
	# 	mse_all_noise = np.array(mse_all_noise)
	# 	# mse_all_noise_mean = np.mean(mse_all_noise, axis=0)
	# 	mse_all_noise_mean = np.min(mse_all_noise, axis=0)
	# 	OUT_MSE[action] = mse_all_noise_mean
	# 	OUT_MSE_A.append(mse_all_noise_mean[ITEMS])

	if config.dataset_type == 'Euler':
		fname_input = join(file_root_euler, action + '_gt.npy')
		fname_pred = join(file_root_euler, action + '_pred.npy')
		inp_euler = np.concatenate((X_Euler_all_noise, Y_Euler_all_noise), axis=2)
		np.save(fname_input, inp_euler)
		np.save(fname_pred, Y_Euler_hat_all_noise)

		fname_input = join(file_root_3D, action + '_gt.npy')
		fname_pred = join(file_root_3D, action + '_pred.npy')
		inp = np.concatenate((X_all_noise, Y_all_noise), axis=2)
		np.save(fname_input, inp)
		np.save(fname_pred, Y_hat_all_noise)

	elif config.dataset_type == '3D':
		fname_input = join(file_root, action + '_gt.npy')
		fname_pred = join(file_root, action + '_pred.npy')
		inp = np.concatenate((X_all_noise, Y_all_noise), axis=2)
		np.save(fname_input, inp)
		np.save(fname_pred, Y_hat_all_noise)



# print('  80   160   320   400   560   640   720   1000   2000   3000   4000')
# for action, metric in OUT_EUC.items():
# 	print(action)
# 	print(np.round(metric[ITEMS] * 1000, 1))
#
# if config.dataset_type == 'Euler':
# 	for action, metric in OUT_MSE.items():
# 		print(action)
# 		print(np.round(metric[ITEMS], 3))
#
# print('\n Average Joints error ')
# OUT_EUC_A_np = np.array(OUT_EUC_A)
# print('  80   160   320   400   560   640   720   1000   2000   3000   4000')
# print(np.round(np.mean(OUT_EUC_A_np, axis=0) * 1000, 1))
#
#
# if config.dataset_type == 'Euler':
# 	print('\n Average Angular error ')
# 	OUT_MSE_A_np = np.array(OUT_MSE_A)
# 	print('  80   160   320   400   560   640   720   1000   2000   3000   4000')
# 	print(np.round(np.mean(OUT_MSE_A_np, axis=0), 3))

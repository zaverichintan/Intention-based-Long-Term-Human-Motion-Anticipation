from torch.utils.data import DataLoader
import numpy as np
import numpy.random as rnd
import mocap.datasets.h36m as H36M
from mocap.mlutil.sequence import PoseDataset
from mocap.datasets.cmu_eval import CMUEval, CMUEval3D, ACTIVITES, DataType, remove_duplicate_joints, \
	recover_duplicate_joints

from ourgan.config import config
from mocap.datasets.custom_activities import CustomActivities
from time import sleep
import mocap.evaluation.h36m as eval_mocap_h36
from ourgan.data_loader.evaluation_angles_auto_regress import get

def wrapper_func_h36_clustered(ds):
	return CustomActivities(ds,
		n_activities=config.label_dim,
		activity_dir= config.activity_dir+'h36m/h36m_clusters' + str(config.label_dim))

def wrapper_func_h36_low_clustered(ds):
	ds = H36M.H36M_Simplified(ds)
	return CustomActivities(ds,
		n_activities=config.label_dim,
		activity_dir= config.activity_dir+'h36m/h36m_clusters' + str(config.label_dim))

def wrapper_func__h36_naive_cluster(ds):
	return CustomActivities(ds,
	                        n_activities=config.label_dim,
	                        activity_dir=config.activity_dir + 'h36m/h36m_naiveclusters' + str(config.label_dim))

def get_CMU_sequences(ds, n_frames, label, num_seeds):
	ds_train_cmu_with_clusters = CustomActivities(ds,
			                           activity_dir=config.activity_dir+'cmu_eval/cmu_clusters8',
			                           n_activities=config.label_dim,
			                           key_as_dir_depth=4)
	dataset = PoseDataset(ds_train_cmu_with_clusters,
	                           n_frames=n_frames,
	                           framerates=config.framrates,
	                           add_noise=True,
	                           noise_var=0.001)
	rnd.seed(config.random_seed)
	print(len(dataset))

	sequence_ids = rnd.choice(a=len(dataset), size=num_seeds, replace=False)

	X = np.empty((num_seeds, n_frames, config.data_dim), dtype=np.float32)
	Y = np.empty((num_seeds, n_frames, config.label_dim), dtype=np.float32)

	for j, i in enumerate(sequence_ids):
		seq, labels = dataset[i]
		X[j] = seq
		Y[j] = labels
	return X, Y
	
def get_H36M_sequences(ds, n_frames, label, num_seeds):
	if config.labels_type == 'clustered':
		ds = CustomActivities(ds, n_activities=config.label_dim,
		                           activity_dir=config.activity_dir + 'h36m/h36m_clusters' + str(config.label_dim))

	elif config.labels_type == 'naive_clustered':
		ds = CustomActivities(ds, n_activities=config.label_dim,
		                           activity_dir=config.activity_dir + 'h36m/h36m_naiveclusters' + str(
			                           config.label_dim))

	dataset = PoseDataset(ds,
	                           n_frames=n_frames,
	                           framerates=config.framrates,
	                           add_noise=True,
	                           noise_var=0.001)
	rnd.seed(config.random_seed)
	print(len(dataset))

	sequence_ids = rnd.choice(a=len(dataset), size=num_seeds, replace=False)

	X = np.empty((num_seeds, n_frames, config.data_dim), dtype=np.float32)
	Y = np.empty((num_seeds, n_frames, config.label_dim), dtype=np.float32)

	for j, i in enumerate(sequence_ids):
		seq, labels = dataset[i]
		X[j] = seq
		Y[j] = labels
	return X, Y

def load_vis_data(action, num_seeds):
	print('...use z-aligned data...')
	sleep(2)
	if config.dataset_name == 'H36M':
		if config.dataset_type == '3D':
			config.data_dim = 96
			if config.labels_type == 'simplified' or config.labels_type == 'zero':
				Seqs, Labels = eval_mocap_h36.get(action, H36M.H36M_FixedSkeleton_withSimplifiedActivities, actor='S5',
				                                  Wrapper_class=None, Wrapper_fn=None, num_seeds=num_seeds,
				                                  data_cbc=None)
			if config.labels_type == 'clustered':
				Seqs, Labels = eval_mocap_h36.get(action, H36M.H36M_FixedSkeleton, actor='S5',
				                                  Wrapper_class=None, Wrapper_fn=wrapper_func_h36_clustered, num_seeds=num_seeds, data_cbc=None)
			elif config.labels_type == 'naive_clustered':
				Seqs, Labels = eval_mocap_h36.get(action, H36M.H36M_FixedSkeleton, actor='S5',
				                                  Wrapper_class=None, Wrapper_fn=wrapper_func__h36_naive_cluster,
				                                  num_seeds=num_seeds, data_cbc=None)

		if config.dataset_type == '3D' and config.long_term_inference:
			config.data_dim = 96
			n_frames = config.n_in + config.n_out*(config.auto_regress_vis+1) + config.forecast
			if config.long_term_inference:
				n_frames = config.n_in + config.n_out * (config.long_term_auto_regress_vis + 1)  + config.forecast

			if config.labels_type == 'simplified' or config.labels_type == 'zero':
				ds = H36M.H36M_FixedSkeleton_withSimplifiedActivities(actions=[action], remove_global_Rt=True, actors=['S5'])
				Seqs, Labels = get_H36M_sequences(ds, n_frames, config.labels_type, num_seeds)

			if config.labels_type == 'clustered':
				ds = H36M.H36M_FixedSkeleton(actions=[action], remove_global_Rt=True, actors=['S5'])
				Seqs, Labels = get_H36M_sequences(ds, n_frames, config.labels_type, num_seeds)

			elif config.labels_type == 'naive_clustered':
				ds = H36M.H36M_FixedSkeleton(actions=[action], remove_global_Rt=True, actors=['S5'])
				Seqs, Labels = get_H36M_sequences(ds, n_frames, config.labels_type, num_seeds)

		if config.dataset_type == 'Euler':
			config.data_dim = 99

			if config.labels_type == 'simplified' or config.labels_type == 'zero':
				Seqs, Labels = eval_mocap_h36.get(action, H36M.H36M_Exp_withSimplifiedActivities, actor='S5',
				                                  Wrapper_class=None, Wrapper_fn=None, num_seeds=num_seeds,
				                                  data_cbc=None, remove_global_Rt=False)
			if config.labels_type == 'clustered':
				Seqs, Labels = eval_mocap_h36.get(action, H36M.H36M_Exp, actor='S5',
				                                  Wrapper_class=None, Wrapper_fn=wrapper_func_h36_clustered, num_seeds=num_seeds, data_cbc=None,remove_global_Rt=False)
			elif config.labels_type == 'naive_clustered':
				Seqs, Labels = eval_mocap_h36.get(action, H36M.H36M_Exp, actor='S5',
				                                  Wrapper_class=None, Wrapper_fn=wrapper_func__h36_naive_cluster,
				                                  num_seeds=num_seeds, data_cbc=None, remove_global_Rt=False)

		if config.dataset_type == 'Euler' and config.long_term_inference:
			config.data_dim = 99
			n_frames = config.n_in + config.n_out*(config.auto_regress_vis+1) + config.forecast
			if config.long_term_inference:
				n_frames = config.n_in + config.n_out * (config.long_term_auto_regress_vis + 1) + config.forecast

			if config.labels_type == 'simplified' or config.labels_type == 'zero':
				ds = H36M.H36M_Exp_withSimplifiedActivities(actions=[action], actors=['S5'])
				Seqs, Labels = get_H36M_sequences(ds, n_frames, config.labels_type, num_seeds)

			if config.labels_type == 'clustered':
				ds = H36M.H36M_Exp(actions=[action], actors=['S5'])
				Seqs, Labels = get_H36M_sequences(ds, n_frames, config.labels_type, num_seeds)

			elif config.labels_type == 'naive_clustered':
				ds = H36M.H36M_Exp(actions=[action], actors=['S5'])
				Seqs, Labels = get_H36M_sequences(ds, n_frames, config.labels_type, num_seeds)

	if config.dataset_name == "H36M_less_joints":
		if config.dataset_type == '3D':
			config.data_dim = 51
			
			if config.labels_type == 'clustered':
				# Seqs, Labels = eval_mocap_h36.get(action, H36M.H36M_FixedSkeletonFromRotation, actor='S5',
				# Seqs, Labels = eval_mocap_h36.get(action, H36M.H36M, actor='S5',
				#                                   Wrapper_class=None, Wrapper_fn=wrapper_func_h36_low_clustered, num_seeds=num_seeds, data_cbc=None, remove_global_Rt=False, remove_global_t=True)
	
				Seqs, Labels = eval_mocap_h36.get(action, H36M.H36M, actor='S5',
				                                  Wrapper_class=None, Wrapper_fn=wrapper_func_h36_low_clustered, num_seeds=num_seeds, data_cbc=None, remove_global_Rt=True)
				Seqs = Seqs.reshape(Seqs.shape[0], Seqs.shape[1], config.data_dim)
				
	if config.dataset_name == "H36M_less_joints_fixed":
		if config.dataset_type == '3D':
			config.data_dim = 51
			
			if config.labels_type == 'clustered':
				# Seqs, Labels = eval_mocap_h36.get(action, H36M.H36M_FixedSkeletonFromRotation, actor='S5',
				Seqs, Labels = eval_mocap_h36.get(action, H36M.H36M_FixedSkeleton, actor='S5',
				                                  Wrapper_class=None, Wrapper_fn=wrapper_func_h36_low_clustered, num_seeds=num_seeds, data_cbc=None, remove_global_Rt=False, remove_global_t=True)
	
	# CMU dataset
	if config.dataset_name == 'CMU':
		if config.dataset_type == '3D':
			config.data_dim = 114
			ds = CMUEval3D([action], DataType.TEST, remove_global_Rt=True)

		if config.dataset_type == 'Euler':
			config.data_dim = 117
			ds = CMUEval([action], DataType.TEST)

		# Add the labels to the dataset - Works for both the 3D and the Euler
		if config.labels_type == 'clustered' or config.labels_type == 'zero':
			n_frames = config.n_in + config.n_out*(config.auto_regress_vis+1)

			if config.long_term_inference:
				n_frames = config.n_in + config.n_out * (config.long_term_auto_regress_vis + 1)
				n_frames = config.n_in + config.n_out*(6)


			Seqs, Labels = get_CMU_sequences(ds, n_frames, config.labels_type, num_seeds)

		elif config.labels_type == 'naive_clustered' or config.labels_type == 'simplified':
			print('Simplified and naive clusters are not implemented')
			exit()

	return Seqs, Labels

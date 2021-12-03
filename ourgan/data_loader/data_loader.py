from torch.utils.data import DataLoader
import mocap.datasets.h36m as H36M
from mocap.mlutil.sequence import PoseDataset
from mocap.datasets.cmu_eval import CMUEval, CMUEval3D, ACTIVITES, DataType, remove_duplicate_joints, \
	recover_duplicate_joints
from mocap.datasets.custom_activities import CustomActivities
from time import sleep
from ourgan.config import config

from mocap.datasets.combined import Combined
from mocap.datasets.amass import AMASS
import mocap.datasets.amass_constants.train as AMASS_TRAIN
import mocap.datasets.amass_constants.test as AMASS_TEST

from utils.zero_activities import ZeroActivities

def load_train_test_data():
	print('...use z-aligned data...')
	sleep(2)
	if config.dataset_name == 'H36M':
		if config.dataset_type == '3D':
			config.data_dim = 96
			ds_train = H36M.H36M_FixedSkeleton(remove_global_Rt=True, actors=['S1', 'S6', 'S7', 'S8', 'S9', 'S11'])
			ds_test = H36M.H36M_FixedSkeleton(remove_global_Rt=True, actors=['S5'])

			if config.labels_type == 'simplified' or config.labels_type == 'zero':
				ds_train = H36M.H36M_FixedSkeleton_withSimplifiedActivities(remove_global_Rt=True,
				                                                            actors=['S1', 'S6', 'S7', 'S8', 'S9',
				                                                                    'S11'])
				ds_test = H36M.H36M_FixedSkeleton_withSimplifiedActivities(remove_global_Rt=True, actors=['S5'])

		if config.dataset_type == 'Euler':
			config.data_dim = 99
			ds_train = H36M.H36M_Exp(actors=['S1', 'S6', 'S7', 'S8', 'S9', 'S11'])
			ds_test = H36M.H36M_Exp(actors=['S5'])

			if config.labels_type == 'simplified' or config.labels_type == 'zero':
				ds_train = H36M.H36M_Exp_withSimplifiedActivities(actors=['S1', 'S6', 'S7', 'S8', 'S9', 'S11'])
				ds_test = H36M.H36M_Exp_withSimplifiedActivities(actors=['S5'])

		# Add the labels to the dataset - Works for both the 3D and the Euler
		if config.labels_type == 'clustered':
			ds_train = CustomActivities(ds_train, n_activities=config.label_dim,
			                            activity_dir=config.activity_dir + 'h36m/h36m_clusters' + str(config.label_dim))
			ds_test = CustomActivities(ds_test, n_activities=config.label_dim,
			                           activity_dir=config.activity_dir + 'h36m/h36m_clusters' + str(config.label_dim))

		elif config.labels_type == 'naive_clustered':
			ds_train = CustomActivities(ds_train, n_activities=config.label_dim,
			                            activity_dir=config.activity_dir + 'h36m/h36m_naiveclusters' + str(
				                            config.label_dim))
			ds_test = CustomActivities(ds_test, n_activities=config.label_dim,
			                           activity_dir=config.activity_dir + 'h36m/h36m_naiveclusters' + str(
				                           config.label_dim))
	
	if config.dataset_name == 'H36M_less_joints_fixed' or config.dataset_name == 'H36M_less_joints':
		if config.dataset_type == '3D' and config.dataset_name == 'H36M_less_joints_fixed':
			config.data_dim = 51
			ds_train = H36M.H36M_FixedSkeleton(
				remove_global_t=True,
				actors=['S1', 'S6', 'S7', 'S8', 'S9', 'S11'])

			ds_test = H36M.H36M_FixedSkeleton(
				remove_global_t=True,
				actors=['S5'])

			ds_train = H36M.H36M_Simplified(ds_train)
			ds_test = H36M.H36M_Simplified(ds_test)

		if config.dataset_type == '3D' and config.dataset_name == 'H36M_less_joints':
			config.data_dim = 51
			ds_train = H36M.H36M(
				remove_global_t=True,
				actors=['S1', 'S6', 'S7', 'S8', 'S9', 'S11'])

			ds_test = H36M.H36M(
				remove_global_t=True,
				actors=['S5'])

			ds_train = H36M.H36M_Simplified(ds_train)
			ds_test = H36M.H36M_Simplified(ds_test)

		# Add the labels to the dataset - Works for both the 3D and the Euler
		if config.labels_type == 'clustered':
			ds_train = CustomActivities(ds_train, n_activities=config.label_dim,
			                            activity_dir=config.activity_dir + 'h36m/h36m_clusters' + str(config.label_dim))
			ds_test = CustomActivities(ds_test, n_activities=config.label_dim,
			                           activity_dir=config.activity_dir + 'h36m/h36m_clusters' + str(config.label_dim))

		elif config.labels_type == 'naive_clustered':
			ds_train = CustomActivities(ds_train, n_activities=config.label_dim,
			                            activity_dir=config.activity_dir + 'h36m/h36m_naiveclusters' + str(
				                            config.label_dim))
			ds_test = CustomActivities(ds_test, n_activities=config.label_dim,
			                           activity_dir=config.activity_dir + 'h36m/h36m_naiveclusters' + str(
				                           config.label_dim))
	
	# CMU dataset
	if config.dataset_name == 'CMU':
		if config.dataset_type == '3D':
			config.data_dim = 114
			ds_train = CMUEval3D(ACTIVITES, DataType.TRAIN, remove_global_Rt=True)
			ds_test = CMUEval3D(ACTIVITES, DataType.TEST, remove_global_Rt=True)

		if config.dataset_type == 'Euler':
			config.data_dim = 117
			ds_train = CMUEval(ACTIVITES, DataType.TRAIN)
			ds_test = CMUEval(ACTIVITES, DataType.TEST)


		# Add the labels to the dataset - Works for both the 3D and the Euler
		if config.labels_type == 'clustered' or config.labels_type == 'zero':
			ds_train = CustomActivities(ds_train,
			                            activity_dir=config.activity_dir+'cmu_eval/cmu_clusters8',
			                            n_activities=config.label_dim,
			                            key_as_dir_depth=4)
			ds_test = CustomActivities(ds_test,
			                           activity_dir=config.activity_dir+'cmu_eval/cmu_clusters8',
			                           n_activities=config.label_dim,
			                           key_as_dir_depth=4)
		elif config.labels_type == 'naive_clustered' or config.labels_type == 'simplified':
			print('Simplified and naive clusters are not implemented')
			exit()

	# Pass the datasets to the pytorch dataloader
	if config.dataset_type == '3D' and config.dataset_name!="AMASS":
		# mirror the data if 3D and from H36M
		dataset_train = PoseDataset(ds_train,
		                                 n_frames=config.n_frames + config.forecast,
		                                 framerates=config.framrates,
		                                 add_noise=True,
		                                 noise_var=0.001,
		                                 mirror_data=True)
	else:
		dataset_train = PoseDataset(ds_train,
		                                 n_frames=config.n_frames + config.forecast,
		                                 framerates=config.framrates,
		                                 add_noise=True,
		                                 noise_var=0.001)

	dataset_test = PoseDataset(ds_test,
	                                n_frames=config.n_frames + config.forecast,
	                                framerates=config.framrates)

	dl_train = DataLoader(dataset_train, batch_size=config.train_batchsize,
	                      shuffle=True, num_workers=8, drop_last=True)
	dl_test = DataLoader(dataset_test, batch_size=config.test_batchsize,
	                     num_workers=8, drop_last=True)

	return dl_train, dl_test

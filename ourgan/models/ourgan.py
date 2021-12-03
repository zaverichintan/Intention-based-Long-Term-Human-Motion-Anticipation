from junn.scaffolding import Scaffolding
import torch.nn as nn
from os.path import join
import torch
from fcgan.models.tcn import TemporalConvNet
import torch.nn.functional as F
from ourgan.config import config

class PoseEncoder(Scaffolding):

	def __init__(self, hidden_units, dim,
	             label_dim, txt, stacks,
	             model_seed, n_in=25, n_out=50,
	             force_new_training=False):
		self.stacks = stacks
		self.hidden_units = hidden_units
		self.n_out = n_out
		self.txt = join(
			'ld' + str(dim) + '_' + str(label_dim) + \
			'_' + str(hidden_units) + '_' + str(stacks) + \
			'_nin' + str(n_in) + '_to_' + str(n_out),
			txt)
		super().__init__(force_new_training=force_new_training,
		                 model_seed=model_seed)

		self.gru = nn.GRU(input_size=dim,
		                  hidden_size=hidden_units,
		                  num_layers=1,
		                  batch_first=True)
		self.pose_encoder = nn.Linear(in_features=hidden_units,
		                              out_features=dim)

	def get_unique_directory(self):
		"""
		"""
		return join('pose_encoder', self.txt)

	def forward(self, pose, h):
		gru_input = pose
		output, h = self.gru(gru_input, h)
		pose = self.pose_encoder(output) + pose
		return pose, h

class PoseDecoder(Scaffolding):

	def __init__(self, hidden_units, dim,
	             label_dim, txt, stacks,
	             model_seed, dropout=0.3, forecast=20, n_in=25, n_out=50,
	             label_embedding=32, force_new_training=False):
		self.stacks = stacks
		self.hidden_units = hidden_units
		self.label_dim = label_dim
		self.forecast = forecast
		self.n_out = n_out
		self.txt = join(
			'ld' + str(dim) + '_' + str(label_dim) + \
			'_' + str(hidden_units) + '_' + str(stacks) + \
			'_nin' + str(n_in) + '_to_' + str(n_out),
			txt)
		super().__init__(force_new_training=force_new_training,
		                 model_seed=model_seed)
		self.decode_labels = nn.Sequential(
			nn.Conv1d(
				in_channels=label_dim,
				out_channels=label_embedding,
				kernel_size=self.forecast
			)
		)


		self.gru = nn.GRU(input_size=dim + label_embedding,
		                  hidden_size=hidden_units,
		                  batch_first=True)
		if config.hidden_units == 512:
			self.gru = nn.GRU(input_size=dim + label_embedding,
			                  hidden_size=hidden_units,
			                  num_layers=stacks,
			                  batch_first=True,
			                  dropout=dropout)

		self.pose_decoder = nn.Linear(in_features=hidden_units,
		                              out_features=dim)

	def get_unique_directory(self):
		"""
		"""
		return join('pose_decoder_', self.txt)

	def forward(self, pose, labels, h):
		labels = labels.permute(0, 2, 1)
		lab = torch.squeeze(self.decode_labels(labels)).unsqueeze(1)
		gru_input = torch.cat([pose, lab], dim=2)
		output, h = self.gru(gru_input, h)
		pose = self.pose_decoder(output) + pose
		return pose, h

class PoseDiscriminator(Scaffolding):

	def __init__(self, hidden_units, dim,
	             label_dim, txt, stacks,
	             model_seed, n_in=25, n_out=50,
	             force_new_training=False):
		self.stacks = stacks
		self.hidden_units = hidden_units
		self.label_dim = label_dim
		self.n_out = n_out
		self.dim = dim
		self.txt = join(
			'ld' + str(dim) + '_' + str(label_dim) + \
			'_' + str(hidden_units) + '_' + str(stacks) + \
			'_nin' + str(n_in) + '_to_' + str(n_out),
			txt)
		super().__init__(force_new_training=force_new_training,
		                 model_seed=model_seed)
		self.first = 512
		self.second = 1

		# -- build model --
		self.fc1 = nn.Linear(dim, self.first)
		self.bn1 = nn.BatchNorm1d(self.first)
		self.fc2 = nn.Linear(self.first, self.second)
		self.bn2 = nn.BatchNorm1d(self.second)

	def get_unique_directory(self):
		"""
		"""
		return join('pose_discriminator_', self.txt)

	def forward(self, poses):
		poses = poses.view(poses.shape[0], self.dim)
		x = self.fc1(poses)
		x = F.relu(self.bn1(x))
		x = self.fc2(x)
		x = F.relu(self.bn2(x))
		# x = self.bn2(x)
		out = torch.sigmoid(x)
		return out

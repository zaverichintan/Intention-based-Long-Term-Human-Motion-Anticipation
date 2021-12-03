from junn.training import Trainer
from ourgan.models.wgan import PoseEncoder, PoseDecoder, PoseDiscriminator
import torch
import torch.nn as nn
import numpy as np
from ourgan.config import config
from torch.autograd import Variable

def euclidean_distance(y_true, y_pred):
	y_true = torch.reshape(y_true, (-1, 3))
	y_true = torch.reshape(y_true, (-1, 3))
	y_pred = torch.reshape(y_pred, (-1, 3))
	dif = y_true - y_pred
	distance = torch.norm(dif + 0.00000001, dim=1)
	return torch.mean(distance)

def calculate_accuracy(Labels_as_classes, Labels_hat):
    """
    :param Labels_as_classes: {n_batch x n_frames}
    :param Labels_hat: {n_batch x n_frames x n_classes}
    """
    n_batch = Labels_as_classes.size(0)
    n_frames = Labels_as_classes.size(1)
    _, Labels_hat_as_classes = torch.max(Labels_hat.data, 2)
    correct = (Labels_hat_as_classes.float() == Labels_as_classes.float()).sum()
    return (correct / float(n_frames * n_batch)).item()

def MSE_loss(y_pred, y_true):
    diff = y_true - y_pred[:,0,:]
    err = torch.mean(torch.sqrt(torch.sum((diff) ** 2, axis=1)), axis=0)
    return err

class Wgan_trainer(Trainer):

	def __init__(self,  hidden_units, device,
                 label_dim, txt, stacks, dim, n_in, n_out,
                 model_seed, force_new_training=False):
		"""
		"""
		self.n_in = n_in
		self.n_out = n_out
		self.stacks = stacks

		# self.CE = nn.CrossEntropyLoss()
		self.CE = nn.BCELoss()
		encoder = PoseEncoder(
			n_in=n_in, n_out=n_out,
			hidden_units=hidden_units, label_dim=label_dim, txt=txt, stacks=stacks,
			model_seed=model_seed, force_new_training=force_new_training, dim=dim
		)
		decoder = PoseDecoder(
			n_in=n_in, n_out=n_out,
			hidden_units=hidden_units + config.noise_dim, label_dim=label_dim, txt=txt, stacks=stacks,
			model_seed=model_seed, force_new_training=force_new_training, forecast=config.forecast, dim=dim
		)
		discriminator = PoseDiscriminator(
			n_in=n_in, n_out=n_out,
			hidden_units=hidden_units, label_dim=label_dim, txt=txt, stacks=stacks,
			model_seed=model_seed, force_new_training=force_new_training, dim=n_out *  config.data_dim
		)

		super().__init__(
			[encoder, decoder, discriminator], put_all_models_in_common_dir=False,
			save_only_best_model=False, max_epoch=config.MAX_EPOCH, device=device,
		)

	def loss_names(self):
		return ['loss_combined', 'pose_loss']

	def train_step(self, epoch, Data, optims):
		for optim in optims:
			optim.zero_grad()
		optimizerEnc = optims[0]
		optimizerDec = optims[1]
		optimizerDisc = optims[2]

		Seq, Labels = Data
		loss_combined, pose_loss = self.step(epoch, Seq, Labels, optimizerEnc, optimizerDec, optimizerDisc)

		return loss_combined.item(), pose_loss.item()

	def val_step(self, epoch, Data):
		Seq, Labels = Data
		with torch.no_grad():
			loss_combined, pose_loss = self.step(epoch, Seq, Labels, optimizerEnc=None, optimizerDec=None, optimizerDisc=None)

		return loss_combined.item(), pose_loss.item()

	def step(self, epoch, Seq, Labels, optimizerEnc, optimizerDec, optimizerDisc):
		"""

		:param epoch:
		:param Seq:
		:param Labels:
		:param optimizerEnc:
		:param optimizerDec:
		:param optimizerDisc:
		:return:
		"""
		n_in = self.n_in
		n_out = self.n_out
		n_frames = n_in + n_out
		stacks = self.stacks
		device = self.device
		one = torch.tensor(1, dtype=torch.float)
		mone = one * -1
		one = one.to(config.device)
		mone = mone.to(config.device)
		real_label = 1
		fake_label = 0

		Wasserstein_D = 0

		is_train_step = optimizerEnc is not None

		criterion = self.CE
		if (config.dataset_type == '3D'):
			loss_function = euclidean_distance
		else:
			loss_function = MSE_loss
		loss = 0
		Seq = Seq.to(device)
		X = Seq[:, :n_in, :]
		Y = Seq[:, n_in:n_frames, :]

		Labels = Labels.to(device)
		# Labels = torch.argmax(Labels, dim=2)
		Labels_X = Labels[:, :n_in]
		Labels_Y = Labels[:, n_in:]

		pose_generator_encoder = self.models[0]
		pose_generator_decoder = self.models[1]
		pose_descriminator = self.models[2]


		batch_size = X.size()[0]
		label = torch.full((batch_size, 1), real_label, device=config.device)
		# Forward pass real batch through D
		output = pose_descriminator(Y)
		d_loss_real = Variable(output.mean(), requires_grad=True)
		if is_train_step:
			d_loss_real.backward(one)

		# Train with fake images
		if is_train_step:
			batch_size = config.train_batchsize
		else:
			batch_size = config.test_batchsize

		noise = torch.randn(1, batch_size, config.noise_dim, device=config.device)
		hidden_gen = torch.zeros((1, batch_size, pose_generator_encoder.hidden_units), device=config.device)


		# Generate fake data
		# Generate fake data
		for frame in range(config.n_in):
			pose = X[:, frame, :].unsqueeze_(1).clone()
			_, hidden_gen = pose_generator_encoder(pose, hidden_gen)

		out2 = torch.zeros((batch_size, config.n_out, config.data_dim), device=config.device)
		hidden_with_noise_label = torch.cat((hidden_gen, noise), 2)

		# warmup
		for frame in range(config.n_in - config.warmup_frames, config.n_in - 1):
			pose = X[:, frame, :].unsqueeze_(1).clone()
			labels_dec = Labels[:, frame:frame + config.forecast, :]

			_, hidden_with_noise_label = pose_generator_decoder(pose, labels_dec, hidden_with_noise_label)
		pose = X[:, -1, :].unsqueeze_(1).clone()

		for frame in range(config.n_out):
			labels_dec = Labels_Y[:, frame:frame + config.forecast, :]
			prediction, hidden_with_noise_label = pose_generator_decoder(pose, labels_dec, hidden_with_noise_label)
			target = Y[:, frame, :]
			if (frame < config.end_frame_decoder):
				factor = (config.n_out + 1 - frame) / config.n_out
			else:
				factor = 0
			loss += factor * loss_function(prediction, target)
			pose = prediction
			out2[:, frame, :] = torch.squeeze(pose.detach())
		fake_poses = out2
		label.fill_(fake_label)
		# Classify all fake batch with D
		output = pose_descriminator(fake_poses)
		# Calculate D's loss on the all-fake batch
		# d_loss_fake = criterion(output, label)
		d_loss_fake = Variable(output.mean(), requires_grad=True)
		if is_train_step:
			d_loss_fake.backward(mone)

		# Train with gradient penalty

		d_loss = d_loss_fake - d_loss_real
		Wasserstein_D = d_loss_real - d_loss_fake
		if is_train_step:
			optimizerDisc.step()

			# Generator update
			for p in pose_descriminator.parameters():
				p.requires_grad = False  # to avoid computation

		# train generator
		# compute loss with fake images
		# Train with fake images
		batch_size = X.size()[0]
		noise = torch.randn(1, batch_size, config.noise_dim, device=config.device)
		hidden = torch.zeros((1, batch_size, pose_generator_encoder.hidden_units), device=config.device)
		hidden_with_noise = torch.zeros((1, batch_size, pose_generator_encoder.hidden_units + config.noise_dim),
		                                device=config.device)

		# Generate fake data

		pose_generator_encoder.zero_grad()
		pose_generator_decoder.zero_grad()

		loss = 0
		# Generate fake data
		for frame in range(config.n_in):
			pose = X[:, frame, :].unsqueeze_(1).clone()
			_, hidden_gen = pose_generator_encoder(pose, hidden_gen)

		out2 = torch.zeros((batch_size, config.n_out, config.data_dim), device=config.device)
		hidden_with_noise_label = torch.cat((hidden_gen, noise), 2)

		# warmup
		for frame in range(config.n_in - config.warmup_frames, config.n_in - 1):
			pose = X[:, frame, :].unsqueeze_(1).clone()
			labels_dec = Labels[:, frame:frame + config.forecast, :]

			_, hidden_with_noise_label = pose_generator_decoder(pose, labels_dec, hidden_with_noise_label)
		pose = X[:, -1, :].unsqueeze_(1).clone()

		for frame in range(config.n_out):
			labels_dec = Labels_Y[:, frame:frame + config.forecast, :]
			prediction, hidden_with_noise_label = pose_generator_decoder(pose, labels_dec, hidden_with_noise_label)
			target = Y[:, frame, :]
			if (frame < config.end_frame_decoder):
				factor = (config.n_out + 1 - frame) / config.n_out
			else:
				factor = 0
			loss += factor * loss_function(prediction, target)
			pose = prediction
			out2[:, frame, :] = torch.squeeze(pose.detach())


		fake_poses = out2

		label.fill_(real_label)  # fake labels are real for generator cost
		output = pose_descriminator(fake_poses)
		g_loss = output.mean()

		errG_combined = g_loss + 10 * loss

		torch.nn.utils.clip_grad_norm_(pose_generator_encoder.parameters(), config.weight_clipping_term)
		torch.nn.utils.clip_grad_norm_(pose_generator_decoder.parameters(), config.weight_clipping_term)

		# Update G
		if is_train_step:
			optimizerEnc.step()
			optimizerDec.step()

		return errG_combined, loss

	def predict(self, X, Y_labels, noise):
		"""
		:param X: [num_seeds, n_out, data_dim]
		:param Y_labels:[num_seeds, n_out, label_dim]
		:param noise:[1, num_seeds, noise_dim]
		:return:
		"""
		if isinstance(X, np.ndarray):
			X = torch.tensor(X, device=config.device).float()
		if isinstance(Y_labels, np.ndarray):
			Y_labels = torch.tensor(Y_labels, device=config.device).float()

		with torch.no_grad():
			X = X.to(config.device)
			Y_labels = Y_labels.to(config.device)
			Labels = Y_labels
			batch_size = X.size(0)
			pose_generator_encoder = self.models[0]
			pose_generator_encoder.eval()
			pose_generator_decoder = self.models[1]
			pose_generator_decoder.eval()

			hidden_gen = torch.zeros((1, batch_size, pose_generator_encoder.hidden_units), device=config.device)

			# Generate fake data
			for frame in range(config.n_in):
				pose = X[:, frame, :].unsqueeze_(1).clone()
				_, hidden_gen = pose_generator_encoder(pose, hidden_gen)

			out2 = torch.zeros((batch_size, config.n_out, config.data_dim), device=config.device)
			hidden_with_noise_label = torch.cat((hidden_gen, noise), 2)
			# warmup
			for frame in range(config.n_in - config.warmup_frames, config.n_in - 1):
				pose = X[:, frame, :].unsqueeze_(1).clone()
				labels_dec = Labels[:, frame:frame + config.forecast, :]

				_, hidden_with_noise_label = pose_generator_decoder(pose, labels_dec, hidden_with_noise_label)

			pose = X[:, -1, :].unsqueeze_(1).clone()

			for frame in range(config.n_out):
				labels_dec = Y_labels[:, frame:frame + config.forecast, :]
				prediction, hidden_with_noise_label = pose_generator_decoder(pose, labels_dec, hidden_with_noise_label)
				pose = prediction
				out2[:, frame, :] = torch.squeeze(pose.detach())
			return out2

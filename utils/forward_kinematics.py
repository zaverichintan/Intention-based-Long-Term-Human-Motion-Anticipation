from ourgan.config import config
import mocap.processing.conversion as conv
import mocap.math.fk as fk
import mocap.math.fk_cmueval as fk_cmu
import numpy as np
import torch

def expmap2euler(seq):
    """
    :param seq: {n x 99}
    """
    n_frames, dim = seq.shape

    euler = np.copy(seq)
    for j in range(n_frames):
        for k in np.arange(0, 115, 3):
            idx = [k, k+1, k+2]
            R = fk_cmu.expmap2rotmat(seq[j, idx])
            euler[j, idx] = fk_cmu.rotmat2euler(R.T)
    euler[:, 0:6] = 0
    return euler[:, 3:]

def forward_kinematics(X, Y, Y_hat):
	n_samples = config.num_seeds
	joints = config.data_dim // 3 - 1

	y_hat_joints_per_action = np.zeros((n_samples, Y_hat.shape[1], joints, 3), dtype=np.float32)
	y_joints_per_action = np.zeros((n_samples, Y.shape[1], joints, 3), dtype=np.float32)
	x_joints_per_action = np.zeros((n_samples, X.shape[1], joints, 3), dtype=np.float32)

	y_hat_full = np.zeros((n_samples, Y_hat.shape[1], joints * 3), dtype=np.float32)
	y_full = np.zeros((n_samples, Y.shape[1], joints * 3), dtype=np.float32)
	x_full = np.zeros((n_samples, X.shape[1], joints * 3), dtype=np.float32)

	for i in range(n_samples):
		x = X[i]
		y = Y[i]
		y_hat = Y_hat[i].cpu()

		# Expmap to Euler
		x_full[i] = conv.expmap2euler(x)
		y_full[i] = conv.expmap2euler(y)
		y_hat_full[i] = conv.expmap2euler(y_hat)

		y_hat_joints_per_action[i] = fk.euler_fk(y_hat_full[i])
		x_joints_per_action[i] = fk.euler_fk(x_full[i])
		y_joints_per_action[i] = fk.euler_fk(y_full[i])

	x_joints_per_action_np = x_joints_per_action.reshape((n_samples, X.shape[1], joints * 3))
	y_joints_per_action_np = y_joints_per_action.reshape((n_samples, Y.shape[1], joints * 3))
	y_hat_joints_per_action_np = y_hat_joints_per_action.reshape((n_samples, Y_hat.shape[1], joints * 3))

	return x_joints_per_action_np, y_joints_per_action_np, y_hat_joints_per_action_np, x_full, y_full, y_hat_full
def forward_kinematicsCMU(X, Y, Y_hat):
	n_samples = config.num_seeds
	joints = config.data_dim // 3 - 1

	y_hat_joints_per_action = np.zeros((n_samples, Y_hat.shape[1], joints * 3), dtype=np.float32)
	y_joints_per_action = np.zeros((n_samples, Y.shape[1], joints * 3), dtype=np.float32)
	x_joints_per_action = np.zeros((n_samples, X.shape[1], joints * 3), dtype=np.float32)

	y_hat_full = np.zeros((n_samples, Y_hat.shape[1], joints * 3), dtype=np.float32)
	y_full = np.zeros((n_samples, Y.shape[1], joints * 3), dtype=np.float32)
	x_full = np.zeros((n_samples, X.shape[1], joints * 3), dtype=np.float32)

	for i in range(n_samples):
		x = X[i]
		y = Y[i]
		y_hat = Y_hat[i]
		x[:, :6] = 0
		# y[:, :6] = 0
		y_hat[:, :6] = 0

		# Expmap to Euler
		x_full[i] = expmap2euler(x)
		y_full[i] = expmap2euler(y)
		y_hat_full[i] = expmap2euler(y_hat)
		# remove glabel rotation and translation

		# Euler to 3D coordinates
		y_hat_joints_per_action[i] = fk_cmu.angular2euclidean(y_hat).astype('float32')
		x_joints_per_action[i] = fk_cmu.angular2euclidean(x).astype('float32')
		y_joints_per_action[i] = fk_cmu.angular2euclidean(y).astype('float32')

	x_joints_per_action_np = x_joints_per_action.reshape((n_samples, X.shape[1], joints * 3))
	y_joints_per_action_np = y_joints_per_action.reshape((n_samples, Y.shape[1], joints * 3))
	y_hat_joints_per_action_np = y_hat_joints_per_action.reshape((n_samples, Y_hat.shape[1], joints * 3))

	return x_joints_per_action_np, y_joints_per_action_np, y_hat_joints_per_action_np,  x_full, y_full, y_hat_full
def forward_kinematics_numpy(X, Y, Y_hat):
	n_samples = config.num_seeds
	joints = config.data_dim // 3 - 1

	y_hat_joints_per_action = np.zeros((n_samples, Y_hat.shape[1], joints, 3), dtype=np.float32)
	y_joints_per_action = np.zeros((n_samples, Y.shape[1], joints, 3), dtype=np.float32)
	x_joints_per_action = np.zeros((n_samples, X.shape[1], joints, 3), dtype=np.float32)

	y_hat_full = np.zeros((n_samples, Y_hat.shape[1], joints * 3), dtype=np.float32)
	y_full = np.zeros((n_samples, Y.shape[1], joints * 3), dtype=np.float32)
	x_full = np.zeros((n_samples, X.shape[1], joints * 3), dtype=np.float32)

	for i in range(n_samples):
		x = X[i]
		y = Y[i]
		y_hat = Y_hat[i]

		# Expmap to Euler
		x_full[i] = conv.expmap2euler(x)
		y_full[i] = conv.expmap2euler(y)
		y_hat_full[i] = conv.expmap2euler(y_hat)

		y_hat_joints_per_action[i] = fk.euler_fk(y_hat_full[i])
		x_joints_per_action[i] = fk.euler_fk(x_full[i])
		y_joints_per_action[i] = fk.euler_fk(y_full[i])

	x_joints_per_action_np = x_joints_per_action.reshape((n_samples, X.shape[1], joints * 3))
	y_joints_per_action_np = y_joints_per_action.reshape((n_samples, Y.shape[1], joints * 3))
	y_hat_joints_per_action_np = y_hat_joints_per_action.reshape((n_samples, Y_hat.shape[1], joints * 3))

	return x_joints_per_action_np, y_joints_per_action_np, y_hat_joints_per_action_np, x_full, y_full, y_hat_full
def forward_kinematics_expmap_only(X, Y, Y_hat):
	n_samples = config.num_seeds
	joints = config.data_dim // 3 - 1

	y_hat_joints_per_action = np.zeros((n_samples, Y_hat.shape[1], joints, 3), dtype=np.float32)
	y_joints_per_action = np.zeros((n_samples, Y.shape[1], joints, 3), dtype=np.float32)
	x_joints_per_action = np.zeros((n_samples, X.shape[1], joints, 3), dtype=np.float32)

	y_hat_full = np.zeros((n_samples, Y_hat.shape[1], joints * 3), dtype=np.float32)
	y_full = np.zeros((n_samples, Y.shape[1], joints * 3), dtype=np.float32)
	x_full = np.zeros((n_samples, X.shape[1], joints * 3), dtype=np.float32)

	for i in range(n_samples):
		x = X[i]
		y = Y[i]
		y_hat = Y_hat[i].cpu()

		# Expmap to Euler
		x_full[i] = conv.expmap2euler(x)
		y_full[i] = conv.expmap2euler(y)
		y_hat_full[i] = conv.expmap2euler(y_hat)

	x_joints_per_action_np = x_joints_per_action.reshape((n_samples, X.shape[1], joints * 3))
	y_joints_per_action_np = y_joints_per_action.reshape((n_samples, Y.shape[1], joints * 3))
	y_hat_joints_per_action_np = y_hat_joints_per_action.reshape((n_samples, Y_hat.shape[1], joints * 3))

	return x_joints_per_action_np, y_joints_per_action_np, y_hat_joints_per_action_np, x_full, y_full, y_hat_full
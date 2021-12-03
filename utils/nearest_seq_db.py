import sys
sys.path.insert(0, '../')
import torch
import mocap.datasets.h36m as H36M
import mocap_metric.math.distances as DIST
from mocap_metric.database import Database, get_directional_transform_pose_fn, HUMAN36M_LIMBS
import numpy as np
from mocap.math.mirror_h36m import mirror_p3d

def nearest_ground_truth(seq_in, action):

	ds = H36M.H36M_FixedSkeletonFromRotation(
		actors=['S5'], actions=[action], remove_global_Rt=True)
	dist = DIST.directional_distance
	db = Database(ds, kernel_size=10, use_velocity=False,
	              framerate=25, distance_function=dist,
	              transform_pose_fn=None,
	              keep_original_seq=True)

	_, i = db.query(seq_in)
	seqid, frame, is_mirrored = db.Meta[i]
	# assert not is_mirrored
	assert seqid in [0, 1]  # as [2, 3] are the mirrored ones!
	assert len(ds) == 2  # only one actor and ONE action!
	Seq_match = ds[seqid]  # THIS IS {ds}, NOT {db}!!!
	Seq_match = Seq_match[frame:frame + 110]
	if is_mirrored:
		Pts3d = Seq_match.reshape(1 * 110, 32, 3)
		# Pts3d[:, :, (0, 1, 2)] = Pts3d[:, :, (0, 2, 1)]
		Pts3d = mirror_p3d(Pts3d)
		LS = [6, 7, 8, 9, 10, 16, 17, 18, 19, 20, 21, 22, 23]
		RS = [1, 2, 3, 4, 5, 24, 25, 26, 27, 28, 29, 30, 31]
		lr = np.array(LS + RS)
		rl = np.array(RS + LS)
		Pts3d[:, lr] = Pts3d[:, rl]
		Seq_match = Pts3d.reshape(110, 96)

	return Seq_match
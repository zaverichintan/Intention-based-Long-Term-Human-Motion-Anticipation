import mocap.datasets.h36m as H36M
from os.path import isdir, join
from mocap.visualization.sequence import SequenceVisualizer
import numpy as np
import mocap.math.fk as FK
import mocap.processing.normalize as norm
from mocap.math.mirror_h36m import mirror_p3d
import sys
sys.path.insert(0, '../')
from mocap_metric.database import Database, get_directional_transform_pose_fn, HUMAN36M_LIMBS
import mocap_metric.math.distances as DIST
import mocap.processing.normalize as norm
import mocap.processing.conversion as conv
from mocap.math.mirror_h36m import mirror_p3d

action = 'walking'


LOC = '../output/ourgan_zerolabel'
assert isdir(LOC), LOC

def get(action):
    output_dir = '../output/ourgan_zerolabel'
    Seq = np.load(join(output_dir, action + '.npy'))
    n_batch, n_frames, _ = Seq.shape
    Seq = Seq.reshape((n_batch, n_frames, 32, 3))
    LS = [6, 7, 8, 9, 10, 16, 17, 18, 19, 20, 21, 22, 23]
    RS = [1, 2, 3, 4, 5, 24, 25, 26, 27, 28, 29, 30, 31]
    lr = np.array(LS + RS)
    rl = np.array(RS + LS)
    Seq[:, :, lr] = Seq[:, :, rl]
    Seq = Seq.reshape(8, n_frames, 96)
    for i in range(8):
        Seq[i] = norm.remove_rotation_and_translation(Seq[i], j_root=0, j_left=6, j_right=1)
    return Seq

    n_batch, n_frames, dim = seq_euler.shape
    seq_euler = seq_euler.reshape((n_batch * n_frames, dim))
    seq = FK.euler_fk(seq_euler)
    Seq = seq.reshape((n_batch, n_frames, 32, 3))

    LS = [6, 7, 8, 9, 10, 16, 17, 18, 19, 20, 21, 22, 23]
    RS = [1, 2, 3, 4, 5, 24, 25, 26, 27, 28, 29, 30, 31]
    lr = np.array(LS + RS)
    rl = np.array(RS + LS)
    Seq[:, :, lr] = Seq[:, :, rl]

    Seq = Seq.reshape(8, n_frames, 96)
    for i in range(8):
        Seq[i] =  norm.remove_rotation_and_translation(Seq[i], j_root=0, j_left=6, j_right=1)

    return Seq

Seq = get(action)

# --
KERNEL_SIZE = 10
seq0 = Seq[0]
seq0 = seq0[:KERNEL_SIZE]

from mocap_metric.database import Database, get_directional_transform_pose_fn, HUMAN36M_LIMBS
import mocap_metric.math.distances as DIST

ds = H36M.H36M_FixedSkeletonFromRotation(
        actors=['S5'], actions=[action], remove_global_Rt=True)
db = Database(ds, kernel_size=KERNEL_SIZE, use_velocity=False,
            framerate=25, distance_function=DIST.directional_distance,
            transform_pose_fn=None,
            keep_original_seq=True)
_, i = db.query(seq0)
seq1 = db.Orig_Seqs[i].reshape(KERNEL_SIZE, 96)

# --

vis_root = '../output'
vis = SequenceVisualizer(vis_root, 'vis_ourgan_zero_' + action, to_file=True)

vis.plot(seq0, seq1, parallel=True, create_video=True, video_fps=12.5)
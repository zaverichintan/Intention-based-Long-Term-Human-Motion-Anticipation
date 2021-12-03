import mocap.datasets.h36m as H36M
from os.path import isdir, join

import numpy as np
import mocap.math.fk as FK
import mocap.processing.normalize as norm
from mocap.math.mirror_h36m import mirror_p3d
import sys
sys.path.insert(0, '../')
from mocap_metric.database import Database, get_directional_transform_pose_fn, HUMAN36M_LIMBS
import mocap_metric.math.distances as DIST
from ourgan.config import config

sys.path.insert(0, '../')
from mocap.visualization.sequence import SequenceVisualizer
from os.path import join, isdir
from os import makedirs
import shutil

action = 'basketball'

LOC = './../data/stacks3_warmup3/stacks3_warmup_3_CMU/3D'
assert isdir(LOC), LOC

def get(action):
    Pred = np.load(join(LOC, action + '_pred.npy'))
    Gt = np.load(join(LOC, action + '_gt.npy'))
    return Pred, Gt


Pred, Gt = get(action)

print("GT", Gt.shape)
print("Pred", Pred.shape)

vis_root = '../output'
vis = SequenceVisualizer(vis_root, "vis_cmu_" + action, to_file=True)
for i in range(8):
    vis.plot(seq1=Gt[i, 0:10], seq2=Pred[i], create_video=True, video_fps=30)
    vis.plot(seq1=Gt[i, 10:110], seq2=Pred[i, :100], parallel=True, create_video=True, video_fps=30)
    
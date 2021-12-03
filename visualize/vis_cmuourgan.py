from mocap.datasets.cmu_eval import CMUEval3D, ACTIVITES, DataType
from os.path import isdir, join
from mocap.visualization.sequence import SequenceVisualizer
import numpy as np
import mocap.math.fk as FK
import mocap.processing.normalize as norm
from mocap.math.mirror_h36m import mirror_p3d
import sys
sys.path.insert(0, '../')
from mocap_metric.database import Database, get_directional_transform_pose_fn, USEFUL_JOINTS_CMU
import mocap_metric.math.distances as DIST

action = 'basketball'
dist = DIST.directional_distance

def get(action):
    INDIR = '/home/tanke/Dev/CVPR21_forecastgan/data/CMU/3D/clustered'
    Gt = np.load(join(INDIR, action + '_gt.npy'))
    Pred = np.load(join(INDIR, action + '_pred.npy'))
    X = Gt[:, :10]
    Seq = np.concatenate([X, Pred], axis=1)
    return Seq


vis_root = '../output'
vis = SequenceVisualizer(vis_root, 'vis_ourgan_' + action, to_file=True)

KERNEL_SIZE = 10
Seq = get(action)
ds = CMUEval3D(activities=[action], datatype=DataType.TEST, remove_global_Rt=True)
db = Database(ds, kernel_size=KERNEL_SIZE, use_velocity=False,
                framerate=60, distance_function=dist,
                transform_pose_fn=None,
                useful_joints=USEFUL_JOINTS_CMU,
                keep_original_seq=True)

seq = Seq[0, :KERNEL_SIZE]

d, i = db.query(seq)
seq_orig = db.Orig_Seqs[i]
# exit(0)

d2, i2 = db.query(seq_orig)

print('d', d, d2)
exit(0)

# exit(1)
vis.plot(seq1=seq_orig,
         create_video=True, video_fps=12.5, plot_jid=False)
vis.plot(seq1=seq,
         create_video=True, video_fps=12.5, plot_jid=False)
# vis.plot(seq1=seq, seq2=seq_orig, parallel=True, 
#          create_video=True, video_fps=12.5, plot_jid=False)

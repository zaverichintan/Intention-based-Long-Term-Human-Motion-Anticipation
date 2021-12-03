import mocap.datasets.h36m as H36M
from os.path import isdir, join
from mocap.visualization.sequence import SequenceVisualizer
import numpy as np
import mocap.math.fk as FK
import mocap.processing.normalize as norm
import mocap.processing.conversion as conv
from mocap.math.mirror_h36m import mirror_p3d
import sys

sys.path.insert(0, '../')
from mocap_metric.database import Database, get_directional_transform_pose_fn, HUMAN36M_LIMBS
import mocap_metric.math.distances as DIST

action = 'walking'

vis_root = '../output'
vis = SequenceVisualizer(vis_root, 'vis_cnn_' + action, to_file=True)

LOC = './../data/cnn'
assert isdir(LOC), LOC

def get(action):
    # fname_in = join(LOC, 'input/' + action + '.npy')
    # fname_out = join(LOC, 'pred/' + action + '.npy')
    fname_in = join(LOC, action + '_in.npy')
    fname_out = join(LOC, action + '_pred.npy')
    inp = np.squeeze(np.load(fname_in))
    outp = np.squeeze(np.load(fname_out))

    n_batch, n_frames, _ = inp.shape
    inp = FK.euler_fk(conv.expmap2euler(inp.reshape((n_batch * n_frames, -1)))).reshape((n_batch, n_frames, -1))

    n_batch, n_frames, _ = outp.shape
    outp = FK.euler_fk(conv.expmap2euler(outp.reshape((n_batch * n_frames, -1)))).reshape((n_batch, n_frames, -1))

    return inp, outp


X, Y = get(action)

vis.plot(seq1=X[0], seq2=Y[0], create_video=True, video_fps=12.5, plot_jid=False)
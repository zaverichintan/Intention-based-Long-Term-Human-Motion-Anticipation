import sys
sys.path.insert(0, '../')
import torch
import mocap.datasets.h36m as H36M
import mocap.evaluation.h36m as H36M_EV
from mocap.datasets.combined import Combined
import numpy as np
from torch.utils.data import DataLoader
from numpy import random
from time import time
from mocap.mlutil.sequence import PoseDataset
from fcgan.trainer.ae_clustering_trainer import ClusteringTrainer
from mocap.visualization.sequence import SequenceVisualizer


action = 'sittingdown'


model_seed = 0
device = torch.device("cuda")

# PARAMS
# ================================
train_batchsize = 32
test_batchsize = 1024
# --
hidden_units = 128
txt = 'h36m_activity11'
label_dim = 11
dim = 42

Seq = H36M_EV.get(action, H36M.H36M_FixedSkeleton, Wrapper_class=Combined)

Triplets = Seq[:, (0,4,8)]


# TRAINER
# ================================
trainer = ClusteringTrainer(
    hidden_units=hidden_units, dim=dim, model_seed=model_seed,
    device=device, txt=txt, force_new_training=False
)
print("#params:", trainer.prettyprint_number_of_parameters())
assert trainer.are_all_weights_loaded()

model = trainer.models[0]
# model.load_weights_for_epoch(0)
model.load_specific_weights('weights_best.h5')


z_norm, Triplets_hat = trainer.predict(Triplets)

vis_root = '/home/user/visualize'
vis = SequenceVisualizer(vis_root, 'vis_ae_' + action, 
                        to_file=True)


for batch in [0, 20, 100, 150]:
    seq1 = Triplets[batch]
    seq2 = Triplets_hat[batch]
    vis.plot(seq1=seq1, seq2=seq2, parallel=True, 
             create_video=True, video_fps=12.5)
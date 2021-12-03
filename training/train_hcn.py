import sys
sys.path.insert(0, '../')
import torch
import mocap.datasets.h36m as H36M
import numpy as np
from torch.utils.data import DataLoader
from hcn.trainer import HCNTrainer
# from hcn.dataset3drot import PoseActivity3DRot
from hcn.datasetcluster import ClusterDataset


trainer = HCNTrainer(force_new_training=True)

# ds_train = PoseActivity3DRot(H36M.H36M_FixedSkeletonFromRotation(
#         remove_global_Rt=True,
#         actors=['S1', 'S6', 'S7', 'S8', 'S9', 'S11']), 
#         add_noise=True, mirror_data=True)
# ds_test = PoseActivity3DRot(H36M.H36M_FixedSkeletonFromRotation(
#         remove_global_Rt=True,
#         actors=['S5']))
ds_train = ClusterDataset()
ds_test = ClusterDataset(is_test=True)

dl_train = DataLoader(ds_train, batch_size=512,
                      shuffle=True, num_workers=8)

dl_test = DataLoader(ds_test, batch_size=1024,
                     shuffle=False, num_workers=8)

print()
print("#train batches", len(dl_train))
print("#test batches", len(dl_test))

optim = torch.optim.Adam(trainer.models[0].parameters(), lr=0.0005, amsgrad=True, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=.92)

trainer.run(dl_train, dl_test, optim=optim, optim_scheduler=scheduler)

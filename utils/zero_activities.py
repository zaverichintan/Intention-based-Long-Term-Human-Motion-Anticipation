from mocap.datasets.dataset import DataSet
import numpy as np
import numba as nb
from os.path import isdir, isfile, join


class ZeroActivities(DataSet):
    """Requires a directory where labels are stored based on the key"""

    def __init__(self, dataset, n_activities, data_target=0):
        Data = dataset.Data
        Keys = dataset.Keys

        # handle labels
        Labels = []
        for sid in range(len(dataset)):
            seq = Data[data_target][sid]
            labels = np.zeros((len(seq), n_activities), dtype=np.float32)
            Labels.append(labels)

        uid = "ZERO"

        Data.append(Labels)
        super().__init__(
            Data,
            Keys=Keys,
            framerate=dataset.framerate,
            iterate_with_framerate=dataset.iterate_with_framerate,
            iterate_with_keys=dataset.iterate_with_keys,
            j_root=dataset.j_root,
            j_left=dataset.j_left,
            j_right=dataset.j_right,
            n_joints=dataset.n_joints,
            name=dataset.name + "_" + uid,
            mirror_fn=dataset.mirror_fn,
            joints_per_limb=dataset.joints_per_limb,
        )
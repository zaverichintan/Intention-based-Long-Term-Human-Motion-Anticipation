import mocap
import mocap.datasets.h36m as H36M
from sklearn.cluster import KMeans
from mocap.mlutil.sequence import PoseDataset
import numpy as np
from os.path import isfile


ds = PoseDataset(
    H36M.H36M_Simplified(H36M.H36M_FixedSkeletonFromRotation(
        remove_global_Rt=True,
        actors=['S1', 'S6', 'S7', 'S8', 'S9', 'S11'])),
        # actors=['S1'])),
        n_frames=70, framerates=[25], stepsize=10)

print("entries:", len(ds))

cluster_file = '/home/user/CVPR21_forecastgan/data/isclusters.npy'
if isfile(cluster_file):
    pass
else:
    Data = []
    for seq in ds:
        Data.append(np.squeeze(seq).reshape((70*51,)))
    Data = np.array(Data)
    print('Data', Data.shape)
    print('... start kmeans')
    kmeans = KMeans(n_clusters=50, random_state=0).fit(Data)
    print('... end kmeans')
    np.save('/home/user/CVPR21_forecastgan/data/isclusters.npy', kmeans.cluster_centers_)

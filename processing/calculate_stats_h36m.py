import sys
sys.path.insert(0, '../')
import mocap.datasets.h36m as H36M
from fcgan.label_statistic import LabelStatistic
from mocap.datasets.custom_activities import CustomActivities

n_clusters = 11

ds = H36M.H36M_FixedSkeleton(remove_global_Rt=True, actors=['S1'])
ds = CustomActivities(
    ds, n_activities=n_clusters,
    activity_dir='/home/user/visualize/h36m_cluster' + str(n_clusters), 
    postfix='_cluster' + str(n_clusters))

print('NAME', ds.name)

seq, labels = ds[0]
print('seq', seq.shape)
print('labels', labels.shape)

# ds = H36M.H36M_FixedSkeleton_withActivities(
#     remove_global_Rt=True, 
#     actors=['S1', 'S6', 'S7', 'S8', 'S9', 'S11', 'S5'])

# stats = LabelStatistic(ds)

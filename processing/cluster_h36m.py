import sys
sys.path.insert(0, '../')
from fcgan.trainer.ae_clustering_trainer import ClusteringTrainer
import mocap.datasets.h36m as H36M
import mocap.evaluation.h36m as H36M_EV
from mocap.datasets.combined import Combined
import torch
import numpy as np
from tqdm.auto import tqdm
from sklearn.cluster import KMeans
import numpy as np
from os.path import join, isdir
from os import makedirs
import shutil


n_labels = 8

model_seed = 0
device = torch.device("cuda")

vis_root = '/home/user/visualize/h36m_cluster' + str(n_labels)
if isdir(vis_root):
    shutil.rmtree(vis_root)
makedirs(vis_root)

# PARAMS
# ================================
train_batchsize = 32
test_batchsize = 1024
# --
hidden_units = 128
txt = 'h36m_activity11'
dim = 42


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

def get_clusters_from_sequence(seq, trainer):
    """
    :param seq: [n_frames x 42]  @50Hz -> ds to 12.5Hz
    """
    Z = []
    n_frames = len(seq)
    Seq = []
    for t in range(0, n_frames - 8):
        o = seq[t:t+9:4]
        Seq.append(o)
    Seq = np.array(Seq)
    z, _ = trainer.predict(Seq)
    return z


ds_train = Combined(H36M.H36M_FixedSkeleton(
        remove_global_Rt=True,
        actors=['S1', 'S6', 'S7', 'S8', 'S9', 'S11']))
ds_test = Combined(H36M.H36M_FixedSkeleton(
        remove_global_Rt=True,
        actors=['S5']))

Clusters = []
for i in tqdm(range(len(ds_train))):
    Clusters.append(
        get_clusters_from_sequence(ds_train[i], trainer))

X = np.concatenate(Clusters, axis=0)

kmeans = KMeans(n_clusters=n_labels, random_state=0).fit(X)

unique, counts = np.unique(kmeans.labels_, return_counts=True)
print('Q', dict(zip(unique, counts)))

def run(ds, kmeans, trainer):
    for i in tqdm(range(len(ds))):
        key = [str(item) for item in ds.Keys[i]]
        seq = ds[i]
        fname = '_'.join(key) + '_cluster' + str(n_labels) + '.npy'

        clusters = get_clusters_from_sequence(seq, trainer)

        labels = np.zeros(len(seq))
        labels_ = kmeans.predict(clusters)
        labels[:len(labels_)] = labels_
        for t in range(len(labels_), len(labels)):
            labels[t] = labels_[-1]
        np.save(join(vis_root, fname), labels.astype('int64'))

run(ds_test, kmeans, trainer)
run(ds_train, kmeans, trainer)

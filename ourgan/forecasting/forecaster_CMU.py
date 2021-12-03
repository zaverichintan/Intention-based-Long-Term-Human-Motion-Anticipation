import numba as nb
import numpy as np
import torch
from os.path import join, isdir, isfile, abspath
# from fcgan.trainer.forecast_trainer import Forecast_Trainer
from ourgan.trainer.forecast_rnn_trainer import ForecastRNN_Trainer
import pathlib


# @nb.njit(
#     nb.float32[:, :, :](nb.int64[:, :]),
#     nogil=True
# )
# def to_onehot(labels):
#     n_batch, n_frames = labels.shape
#     out = np.zeros((n_batch, n_frames, 8), dtype=np.float32)
#     for b in range(n_batch):
#         for t in range(n_frames):
#             pos = labels[b, t]
#             out[b, t, pos] = 1.0
#     return out


@nb.njit(
    nb.float32[:, :, :](nb.float32[:, :, :]),
    nogil=True
)
def double_size(labels):
    n_batch, n_frames, n_dim = labels.shape
    result = np.zeros((n_batch, n_frames*2, n_dim), dtype=np.float32)
    for t in range(n_frames):
        a = t*2
        b = t*2+1
        result[:, a] = labels[:, t]
        result[:, b] = labels[:, t]
    return result


def __forecast(Seq, requested_n_out, txt='', dim=114):
    """
    :param Seq: [n_batch x 50 x 96]  sequence at 25Hz
    """
    n_in = 25
    n_out = 50
    hidden_units = 12
    hidden_units_disc = 32
    framerate = 12.5
    stacks = 3
    label_dim = 8
    trainer = ForecastRNN_Trainer(
        n_in=n_in, n_out=n_out, hidden_units=hidden_units, 
        device=torch.device("cuda"), label_dim=label_dim, 
        txt=txt, model_seed=0, stacks=stacks,
        hidden_units_disc=hidden_units_disc,
        force_new_training=False, dim=dim)

    local_path = abspath(pathlib.Path(__file__).parent.absolute())
    data_loc = abspath(join(local_path, '../../data/models_cmu'))
    fname = join(data_loc, 'rnnweights' + txt + '.h5')
    assert isfile(fname), fname
    rnn = trainer.models[0]
    rnn.load_specific_weights(fname)
    Labels = trainer.predict(Seq[:, ::2], n_out=requested_n_out//2)
    return double_size(Labels)


def forecast_clusters(Seq, n_out):
    """
    :param Seq: [n_batch x 50 x 114] 3DFixedSkeleton @25Hz
    """
    return __forecast(Seq, requested_n_out=n_out, txt='_cluster_CMU')


def forecast_clusters_exp(Seq, n_out):
    """
    :param Seq: [n_batch x 50 x 117] Euler @25Hz
    """
    return __forecast(Seq, requested_n_out=n_out, txt='_euler_cluster_CMU', dim=117)

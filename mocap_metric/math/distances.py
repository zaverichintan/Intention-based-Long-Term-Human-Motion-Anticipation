import numpy as np
import numba as nb
import numpy.linalg as la
import math as m
from mocap_metric.math.derivatives import velocity


@nb.njit(nb.float32[:](
    nb.float32[:, :, :], nb.float32[:, :, :]
), nogil=True)
def calculate_euclidean_distance(Y, Y_hat):
    """
    :param Y: [n_sequences, n_frames, 96]
    :param Y_hat: [n_sequences, n_frames, 96]
    :return : [n_frames]
    """
    n_sequences, n_frames, dim = Y.shape
    J = dim // 3
    result = np.zeros(shape=(n_frames,), dtype=np.float32)
    for s in range(n_sequences):
        for t in range(n_frames):
            total_euc_error = 0
            for jid in range(0, dim, 3):
                x_gt = Y[s, t, jid]
                y_gt = Y[s, t, jid + 1]
                z_gt = Y[s, t, jid + 2]
                x_pr = Y_hat[s, t, jid]
                y_pr = Y_hat[s, t, jid + 1]
                z_pr = Y_hat[s, t, jid + 2]
                val = (x_gt - x_pr) ** 2 + \
                      (y_gt - y_pr) ** 2 + \
                      (z_gt - z_pr) ** 2
                val = max(val, 0.00000001)
                euc = m.sqrt(val)
                total_euc_error += euc
            result[t] += (total_euc_error / J)
    result = result / n_sequences
    return result


def euclidean_distance(true_seq, query_seq, kernel_size, n_useful_joints):
    """
    :param true_seq: [kernel_size x n_useful_joints*3]
    :param query_seq: [kernel_size x n_useful_joints*3]
    """
    true_seq = np.expand_dims(true_seq, axis=0)
    query_seq = np.expand_dims(query_seq, axis=0)
    output = calculate_euclidean_distance(true_seq, query_seq)
    return np.mean(output)


@nb.njit(nb.float32(
    nb.float32[:, :], nb.float32[:, :], nb.int64, nb.int64
), nogil=True)
def velocity_distance(true_seq, query_seq, kernel_size, n_useful_joints):
    """
    :param true_seq: [kernel_size x n_useful_joints*3]
    :param query_seq: [kernel_size x n_useful_joints*3]
    """
    true_v = np.expand_dims(velocity(true_seq), axis=0)
    query_v = np.expand_dims(velocity(query_seq), axis=0)
    output = calculate_euclidean_distance(true_v, query_v)
    return np.mean(output)


@nb.njit(nb.float32(
    nb.float32[:, :], nb.float32[:, :], nb.int64, nb.int64
), nogil=True)
def directional_distance(true_seq, query_seq, kernel_size, n_useful_joints):
    """
    :param true_seq: [kernel_size x n_useful_joints*3]
    :param query_seq: [kernel_size x n_useful_joints*3]
    """
    true_v_ = np.ascontiguousarray(velocity(true_seq))
    query_v_ = np.ascontiguousarray(velocity(query_seq))
    dim = true_v_.shape[1]
    n_useful_joints = dim // 3

    true_v = true_v_.reshape((kernel_size-1, n_useful_joints, 3))
    query_v = query_v_.reshape((kernel_size-1, n_useful_joints, 3))

    total_score = 0.0
    for t in range(kernel_size-1):
        for jid in range(n_useful_joints):
            a = np.expand_dims(true_v[t,jid], axis=0)
            b_T = np.expand_dims(query_v[t,jid], axis=1)
            norm_a = max(la.norm(a), 0.0000001)
            norm_b = max(la.norm(b_T), 0.0000001)
            cos_sim_ = (a @ b_T) / (norm_a * norm_b)
            cos_sim = cos_sim_[0, 0]
            if cos_sim < 0.9999:
                if np.sum(a[0] - b_T[:, 0]) < 0.000001:
                    cos_sim = 1.0
            # if norm_a < 0.0035 and norm_b < 0.0035:
            #     disp = 1.0
            #     cos_sim = 1.0
            # else:
            disp = min(norm_a, norm_b) / max(norm_a, norm_b)
            # score = ((cos_sim + 1) + disp) / 3.0
            score = ((cos_sim + 1) * disp) / 2.0

            # if score < 1:
            #     print('-------------')
            #     print('norm_b', norm_b)
            #     print('norm_a', norm_a)
            #     print('a', a)
            #     print('b', b_T)
            #     print('disp', disp)
            #     print('cos_sim', cos_sim)
            #     print('score', score)

            #     print('***********')

            # exit(0)

            total_score += (score/n_useful_joints)

    return total_score/(kernel_size-1)


def create_orientation_distance(limbs):

    def orientation_distance(true_seq, query_seq, kernel_size, n_useful_joints):
        """
        :param true_seq: [kernel_size x 4*3]
        :param query_seq: [kernel_size x 4*3]
        """
        true_seq = true_seq.reshape(kernel_size, -1, 3)
        query_seq = query_seq.reshape(kernel_size, -1, 3)

        O = []
        for i in range(4):
            gt = true_seq[:, i, :]
            pred = query_seq[:, i, :]
            for t in range(kernel_size):
                O.append(gt[t] @ pred[t])
        return np.mean(O)

    return orientation_distance
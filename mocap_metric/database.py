import numpy as np
import numba as nb
import math as m
from os.path import join, isfile
from annoy import AnnoyIndex
from mocap_metric.math.derivatives import velocity
from mocap_metric.math.distances import calculate_euclidean_distance
import numpy.linalg as la


USEFUL_JOINTS_H36M = np.array([
    1, 2, 3,
    6, 7, 8,
    25, 26, 27,
    17, 18, 19
]).astype('int64')

USEFUL_JOINTS_CMU = np.array([
    30, 31, 32,
    8, 9, 10,
    21, 22, 23,
    2, 3, 4
]).astype('int64')

USEFUL_JOINTS_MOTIONGAN = np.array([
    1, 2, 3,
    4, 5, 6,
    11, 12, 13,
    14, 15, 16
]).astype('int64')

HUMAN36M_LIMBS = {
    'left_arm': [17, 18, 19],
    'left_leg': [6, 7, 8],
    'right_leg': [1, 2, 3],
    'right_arm': [25, 26, 27]
}

CMU_LIMBS = {
    'left_arm': [30, 31, 32],
    'left_leg': [8, 9, 10],
    'right_leg': [21, 22, 23],
    'right_arm': [2, 3, 4]
}

def get_directional_transform_pose_fn(limbs):
    """
    """

    def get_directions(limb, seq):
        a, b, c = limb
        seq_a = seq[:, a]
        seq_b = seq[:, b]
        seq_c = seq[:, c]
        ab = seq_a - seq_b
        ab_norm = np.expand_dims(la.norm(ab, axis=1), axis=1)
        ab = ab/ab_norm
        bc = seq_b - seq_c
        bc_norm = np.expand_dims(la.norm(bc, axis=1), axis=1)
        bc = bc/bc_norm
        return ab, bc
    
    def transform_pose_fn(seq):
        """
        :param seq: [n_frames x dim]
        """
        n_frames = len(seq)
        seq = seq.reshape((n_frames, -1, 3))
        seq_out = np.empty((n_frames, 8, 3), dtype=np.float32)
        ab, bc = get_directions(limbs['right_arm'], seq)
        seq_out[:, 0] = ab
        seq_out[:, 1] = bc
        ab, bc = get_directions(limbs['left_arm'], seq)
        seq_out[:, 2] = ab
        seq_out[:, 3] = bc
        ab, bc = get_directions(limbs['right_leg'], seq)
        seq_out[:, 4] = ab
        seq_out[:, 5] = bc
        ab, bc = get_directions(limbs['left_leg'], seq)
        seq_out[:, 6] = ab
        seq_out[:, 7] = bc
        return seq_out.reshape((n_frames, -1))

    return transform_pose_fn


class Database:

    def __init__(self, ds, framerate, kernel_size,
                 mirror=True, keep_original_seq=False,
                 use_velocity=False, distance_function=None,
                 useful_joints=None, transform_pose_fn=None,
                 data_dir='/tmp'):
        if transform_pose_fn != None:  # function(seq)
            assert useful_joints is None
        if useful_joints is None:
            useful_joints = USEFUL_JOINTS_H36M
        self.transform_pose_fn = transform_pose_fn
        self.kernel_size = kernel_size
        self.distance_function = distance_function
        self.use_velocity = use_velocity
        self.useful_joints = useful_joints
        self.n_joints = ds.n_joints
        self.n_useful_joints = len(useful_joints)
        self.keep_original_seq = keep_original_seq
        self.transform_pose_fn_dim = -1
        self.Seqs = []
        self.Meta = []
        self.Orig_Seqs = []
        for seqid in range(len(ds)):
            seq = ds[seqid]
            Hrz = ds.get_framerate(seqid)
            self._add_seq(seqid, seq, ds, Hrz, framerate, 
                          keep_original_seq, is_mirrored=False)
            if mirror:
                seq = ds.mirror(seq)
                self._add_seq(seqid, seq, ds, Hrz, framerate, 
                              keep_original_seq, is_mirrored=True)
        if keep_original_seq:
            assert len(self.Seqs) == len(self.Orig_Seqs)
        
        n_dim = len(self.Seqs[0])
        self.lookup = AnnoyIndex(n_dim, 'euclidean')
        for i, v in enumerate(self.Seqs):
            self.lookup.add_item(i, v)
        self.lookup.build(10)
    
    def __len__(self):
        return len(self.Seqs)

    def _add_seq(self, seqid, seq, ds, Hrz, framerate, keep_original_seq, is_mirrored):
        transform_pose_fn = self.transform_pose_fn
        useful_joints = self.useful_joints
        kernel_size = self.kernel_size
        n_frames = len(seq)
        seq = np.reshape(seq, (n_frames, -1, 3))
        if transform_pose_fn is None:
            seq_ = seq[:, useful_joints, :]
        else:
            # seq_ = []
            # for pose in seq:
            #     seq_.append(transform_pose_fn(pose))
            # seq_ = np.array(seq_)
            seq_ = transform_pose_fn(seq)
            if self.transform_pose_fn_dim == -1:
                self.transform_pose_fn_dim = seq_.shape[1]
            else:
                assert self.transform_pose_fn_dim == seq_.shape[1]
        seq_ = np.reshape(seq_, (n_frames, -1))        
        ss = int(round(Hrz/framerate))
        kernel_size_abs = ss * kernel_size
        for t in range(n_frames - kernel_size_abs):
            seq_ss = np.ascontiguousarray((seq_[t:t+kernel_size_abs:ss]))
            if self.use_velocity:
                seq_ss = velocity(seq_ss)
            seq_ss = seq_ss.flatten()
            self.Seqs.append(seq_ss)
            self.Meta.append((seqid, t, is_mirrored))
            if keep_original_seq:
                seq_ss_orig = np.ascontiguousarray(seq[t:t+kernel_size_abs:ss])
                self.Orig_Seqs.append(seq_ss_orig)
        
    def _reduce_to_valid_joints(self, subseq):
        transform_pose_fn = self.transform_pose_fn
        if transform_pose_fn is None:
            n_kernel = len(subseq)
            subseq = np.reshape(subseq, (n_kernel, self.n_joints, 3))
            subseq = subseq[:, self.useful_joints, :]
            subseq = subseq.reshape((n_kernel, -1))
            return np.ascontiguousarray(subseq)
        else:
            return transform_pose_fn(subseq)
    
    def query(self, subseq):
        assert len(subseq) == self.kernel_size
        if self.transform_pose_fn == None and subseq.shape[1] != self.n_useful_joints * 3:
            subseq = self._reduce_to_valid_joints(subseq)
        elif self.transform_pose_fn != None and subseq.shape[1] != self.transform_pose_fn_dim:
            subseq = self._reduce_to_valid_joints(subseq)
        if self.use_velocity:
            subseq = velocity(subseq)
        subseq = subseq.flatten()
        i, dist = self.lookup.get_nns_by_vector(vector=subseq,
                                                n=1, 
                                                include_distances=True)
        i = i[0]
        dist = dist[0]
        true_seq = self.Seqs[i]
        kernel_size = self.kernel_size
        if self.use_velocity:
            kernel_size -= 1
        true_seq = np.reshape(true_seq, (1, kernel_size, -1))
        query_seq = np.reshape(subseq, (1, kernel_size, -1))
   
        if self.distance_function is None:
            d = np.mean(calculate_euclidean_distance(true_seq, query_seq))
        else:
            d = self.distance_function(true_seq[0], query_seq[0], kernel_size, self.n_useful_joints)
        if d < 0.0000001:
            d = 0

        return d, i
    
    def rolling_query(self, subseq):
        assert len(subseq) > self.kernel_size, str(subseq.shape)
        if subseq.shape[1] != self.n_useful_joints * 3:
            subseq = self._reduce_to_valid_joints(subseq)
        n_frames = len(subseq)
        distances = []
        identities = []
        for t in range(n_frames - self.kernel_size):
            d, i = self.query(subseq[t:t+self.kernel_size])
            distances.append(d)
            identities.append(i)
        return distances, identities
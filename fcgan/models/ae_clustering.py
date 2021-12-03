from junn.scaffolding import Scaffolding
import torch.nn as nn
from os.path import join
import torch


class ClusterAE(Scaffolding):

    def __init__(self, hidden_units, dim, model_seed, force_new_training, txt):
        self.hidden_units = hidden_units
        self.dim = dim
        self.txt = txt
        super().__init__(force_new_training=force_new_training,
                         model_seed=model_seed)
        
        self.encoder = nn.Sequential(
            nn.Linear(3*dim, hidden_units),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_units),
            nn.Linear(hidden_units, 16)
        )

        self.decoder = nn.Sequential(
            nn.Linear(16, hidden_units),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_units),
            nn.Linear(hidden_units, 3*dim)
        )
    
    def get_unique_directory(self):
        """
        """
        return join(
            'cluster_ae',
            'h' + str(self.hidden_units) + '_d' + str(self.dim) + '_' + self.txt)
    
    def forward(self, poses):
        """
        :param poses: {n_batch x 3 x dim}
        """
        n_batch = poses.size(0)
        poses = poses.view(n_batch, -1)
        
        z = self.encoder(poses)
        l2norm = torch.norm(z, p=None, dim=1).unsqueeze(1)
        z_norm = z/l2norm
        poses_hat = self.decoder(z_norm).view(n_batch, 3, self.dim)
        return z_norm, z, poses_hat

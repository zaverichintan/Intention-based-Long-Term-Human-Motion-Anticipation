from junn.training import Trainer
from fcgan.models.ae_clustering import ClusterAE
import torch
import torch.nn as nn
import numpy as np


def euclidean_distance(y_true, y_pred):
    y_true = torch.reshape(y_true, (-1, 3))
    y_pred = torch.reshape(y_pred, (-1, 3))
    dif = y_true - y_pred
    distance = torch.norm(dif + 0.00000001, dim=1)
    return torch.mean(distance)


class ClusteringTrainer(Trainer):

    def __init__(self, hidden_units, dim, model_seed, device,
                 txt, force_new_training=False):
        """
        """
        model = ClusterAE(hidden_units, dim, model_seed, force_new_training, txt)
        super().__init__(
            [model], save_only_best_model=False, max_epoch=100, device=device,
        )
    
    def train_step(self, epoch, Seq, optim):
        optim.zero_grad()
        loss, rec, norm = self.step(epoch, Seq)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.models[0].parameters(), 1)
        optim.step()
        return loss.item(), rec, norm
    
    def val_step(self, epoch, Seq):
        loss, rec, norm = self.step(epoch, Seq)
        return loss.item(), rec, norm

    def loss_names(self):
        return ['loss', ' rec', ' norm']
    
    def step(self, epoch, Seq):
        """
        """
        n_batch = Seq.size(0)
        Seq = Seq.to(self.device)

        # -- set upper body to 0
        Seq = Seq.reshape(n_batch, 3, 14, 3)
        upper_body_joints = [6, 7, 8, 9, 10, 11, 12, 13]
        Seq[:, :, upper_body_joints] = 0
        # --

        model = self.models[0]
        z_norm, z, x_hat = model(Seq)

        reconstruction_loss = euclidean_distance(Seq, x_hat)
        norm_loss = torch.mean((torch.norm(z, p=None, dim=1) - 1)**2)

        loss = 0.001 * norm_loss + reconstruction_loss

        return loss, reconstruction_loss.item(), norm_loss.item()
    
    def predict(self, Seq):
        if isinstance(Seq, np.ndarray):
            Seq = torch.from_numpy(Seq)
        Seq = Seq.to(self.device)
        with torch.no_grad():
            model = self.models[0]
            model.eval()

            # -- set upper body to 0
            n_batch = Seq.size(0)
            Seq = Seq.reshape(n_batch, 3, 14, 3)
            upper_body_joints = [6, 7, 8, 9, 10, 11, 12, 13]
            Seq[:, :, upper_body_joints] = 0
            # --

            z_norm, _, x_hat = model(Seq)

            return z_norm.cpu().numpy(), x_hat.cpu().numpy()

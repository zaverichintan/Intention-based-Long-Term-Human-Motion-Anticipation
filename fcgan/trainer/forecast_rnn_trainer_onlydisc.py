from junn.training import Trainer
import torch
import torch.nn as nn
import numpy as np
import numba as nb
from fcgan.models.forecast_rnn import ForecastRNN, Discriminator


@nb.njit(nb.float32[:, :, :](nb.float32[:, :, :]), nogil=True)
def to_onehot(Labels_raw):
    """
    :param Labels: [n_batch x n_frames x 8]
    """
    Labels = np.zeros_like(Labels_raw)
    n_batch, n_frames, dim = Labels_raw.shape
    for batch in range(n_batch):
        for frame in range(n_frames):
            entry = np.argmax(Labels_raw[batch, frame])
            Labels[batch, frame, entry] = 1.0
    return Labels


def calculate_accuracy(Labels_as_classes, Labels_hat):
    """
    :param Labels_as_classes: {n_batch x n_frames}
    :param Labels_hat: {n_batch x n_frames x n_classes}
    """
    n_batch = Labels_as_classes.size(0)
    n_frames = Labels_as_classes.size(1)
    _, Labels_hat_as_classes = torch.max(Labels_hat.data, 2)
    correct = (Labels_hat_as_classes.float() == Labels_as_classes.float()).sum()
    return (correct / float(n_frames * n_batch)).item()


class ForecastRNN_TrainerOnlyDisc(Trainer):

    def __init__(self, hidden_units, hidden_units_disc, 
                 device, stacks,
                 label_dim, txt, dim, n_in, n_out,
                 model_seed, force_new_training=False):
        """ pass
        """
        self.n_in = n_in
        self.n_out = n_out
        self.label_dim = label_dim
        self.criterion = nn.BCELoss()
        txt += '_onlydisc'

        rnn = ForecastRNN(
            hidden_units, dim, label_dim, txt, stacks,
            n_in, n_out, model_seed=model_seed,
            force_new_training=force_new_training)
        
        disc = Discriminator(
            hidden_units=hidden_units_disc, n_out=n_out,
            label_dim=label_dim, txt=txt, model_seed=model_seed,
            force_new_training=force_new_training)

        super().__init__(
            models=[rnn, disc], put_all_models_in_common_dir=False, 
            save_only_best_model=False, max_epoch=100, device=device)
    
    def loss_names(self):
        return ['errG', ' D_x', ' D_G_z1', ' D_G_z2', ' accY']
    
    def train_step(self, epoch, Data, optims):

        optimizerG, optimizerD = optims
        Seq, Labels = Data
        errG, D_x, D_G_z1, D_G_z2, accy = self.step(epoch, Seq, Labels, optimizerG, optimizerD)

        return errG, D_x, D_G_z1, D_G_z2, accy
    
    def val_step(self, epoch, Data):
        Seq, Labels = Data
        errG, D_x, D_G_z1, D_G_z2, accy = self.step(epoch, Seq, Labels, 
                                              optimizerG=None,
                                              optimizerD=None)
        return errG, D_x, D_G_z1, D_G_z2, accy
    
    def step(self, epoch, Seq, Labels, optimizerG, optimizerD):
        
        # Establish convention for real and fake labels during training
        real_label = 1.
        fake_label = 0.

        is_train_step = optimizerG is not None
        n_batch = Seq.size(0)
        n_in = self.n_in
        n_out = self.n_out
        label_dim = self.label_dim
        # CE = self.CE
        device = self.device
        netG, netD = self.models
        criterion = self.criterion
        netD.zero_grad()

        Seq_in = Seq[:, :n_in].cuda()
        Labels = Labels[:, n_in:, ].cuda()
        Labels_class = torch.argmax(Labels, axis=2)

        Labels_hat_raw, Labels_hat = netG(Seq_in)
         
        # long-term loss
        Labels = Labels + torch.randn_like(Labels) * 0.1
        Labels_hat = Labels_hat + torch.randn_like(Labels) * 0.1

        # -- train D --
        label = torch.full((n_batch,), real_label, dtype=torch.float, device=device)
        output = netD(Labels).view(-1)
        if is_train_step:
            errD_real = criterion(output, label)
            errD_real.backward()
        D_x = output.mean().item()
        fake = Labels_hat
        label.fill_(fake_label)
        output = netD(fake.detach()).view(-1)
        if is_train_step:
            errD_fake = criterion(output, label)
            errD_fake.backward()
            optimizerD.step()
        D_G_z1 = output.mean().item()

        # -- train G --
        netG.zero_grad()

        Labels_hat_raw, Labels_hat = netG(Seq_in)
        Labels_hat = Labels_hat + torch.randn_like(Labels) * 0.1

         # short-term loss
        # short_term = 10
        # raw = Labels_hat_raw[:, :short_term].reshape((n_batch * short_term, label_dim))
        # clas = Labels_class[:, :short_term].reshape((n_batch * short_term))
        # loss = CE(raw, clas)
        accy = calculate_accuracy(
            Labels_class.view(n_batch, n_out), 
            Labels_hat_raw.view(n_batch, n_out, self.label_dim))
        fake = Labels_hat
        label.fill_(real_label)
        output = netD(fake).view(-1)
        errG = criterion(output, label)
        if is_train_step:
            errG.backward()
            optimizerG.step()
        D_G_z2 = output.mean().item()

        return errG.item(), D_x, D_G_z1, D_G_z2, accy

    def predict(self, Seq, n_out=0):
        if isinstance(Seq, np.ndarray):
            Seq = torch.from_numpy(Seq)
        n_in = self.n_in
        if n_out == 0:
            n_out = self.n_out
        assert Seq.size(1) == n_in, str(Seq.size())
        with torch.no_grad():
            Seq_in = Seq.cuda()
            rnn = self.models[0].eval()
            _, Labels = rnn(Seq_in, n_out=n_out)
            return to_onehot(Labels.cpu().numpy())
    
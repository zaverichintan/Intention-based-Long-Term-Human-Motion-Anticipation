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


class ForecastRNN_TrainerNoDisc(Trainer):

    def __init__(self, hidden_units, hidden_units_disc, 
                 device, stacks,
                 label_dim, txt, dim, n_in, n_out,
                 model_seed, force_new_training=False):
        """ pass
        """
        self.n_in = n_in
        self.n_out = n_out
        self.label_dim = label_dim
        self.CE = nn.CrossEntropyLoss()
        self.criterion = nn.BCELoss()
        txt += '_nodisc'
        rnn = ForecastRNN(
            hidden_units, dim, label_dim, txt, stacks,
            n_in, n_out, model_seed=model_seed,
            force_new_training=force_new_training)

        super().__init__(
            models=[rnn], put_all_models_in_common_dir=False, 
            save_only_best_model=False, max_epoch=100, device=device)
    
    def loss_names(self):
        return [' accY']
    
    def train_step(self, epoch, Data, optims):
        for optim in optims:
            optim.zero_grad()
        Seq, Labels = Data
        loss, accy = self.step(epoch, Seq, Labels)
        # for m in self.models:
        #     torch.nn.utils.clip_grad_norm_(m.parameters(), 1)
        loss.backward()
        for optim in optims:
            optim.step()
        return accy
    
    def val_step(self, epoch, Data):
        Seq, Labels = Data
        _, accy = self.step(epoch, Seq, Labels)
        return accy
    
    def step(self, epoch, Seq, Labels):
        CE = self.CE
        n_batch = Seq.size(0)
        n_in = self.n_in
        n_out = self.n_out
        label_dim = self.label_dim
        device = self.device
        netG = self.models[0]
        
        Seq_in = Seq[:, :n_in].cuda()
        Labels = Labels[:, n_in:, ].cuda()
        Labels_class = torch.argmax(Labels, axis=2)

        Labels_hat_raw, Labels_hat = netG(Seq_in)
        
         # short-term loss
        short_term = Labels_hat_raw.size(1)
        raw = Labels_hat_raw[:, :short_term].reshape((n_batch * short_term, label_dim))
        clas = Labels_class[:, :short_term].reshape((n_batch * short_term))
        loss = CE(raw, clas)
        accy = calculate_accuracy(
            Labels_class.view(n_batch, n_out), 
            Labels_hat_raw.view(n_batch, n_out, self.label_dim))
        
        return loss, accy

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
    
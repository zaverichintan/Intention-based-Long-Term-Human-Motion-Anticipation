from junn.training import Trainer
from fcgan.models.forecast import SegmentForecastingEncoder, SegmentForecastingDecoder
import torch
import torch.nn as nn
import numpy as np


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


class Forecast_Trainer(Trainer):

    def __init__(self, hidden_units, device, stacks,
                 label_dim, txt, dim, n_in, n_out,
                 model_seed, force_new_training=False):
        """ pass
        """
        self.n_in = n_in
        self.n_out = n_out
        self.label_dim = label_dim
        self.CE = nn.CrossEntropyLoss()

        E = SegmentForecastingEncoder(
            hidden_units=hidden_units, dim=dim, label_dim=label_dim,
            txt=txt, model_seed=model_seed, n_in=n_in, n_out=n_out,
            force_new_training=force_new_training, stacks=stacks)
        D = SegmentForecastingDecoder(
            hidden_units=hidden_units*2, dim=dim, label_dim=label_dim,
            txt=txt, model_seed=model_seed, n_in=n_in, n_out=n_out,
            force_new_training=force_new_training, stacks=stacks)

        super().__init__(
            [E, D], put_all_models_in_common_dir=False, 
            save_only_best_model=True, max_epoch=100, device=device)
        
    def loss_names(self):
        return ['loss', ' accX', ' accY']

    def train_step(self, epoch, Data, optims):
        for optim in optims:
            optim.zero_grad()
        Seq, Labels = Data
        loss, accx, accy = self.step(epoch, Seq, Labels)
        loss.backward()
        for m in self.models:
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1)
        for optim in optims:
            optim.step()
        return loss.item(), accx, accy
    
    def val_step(self, epoch, Data):
        Seq, Labels = Data
        loss, accx, accy = self.step(epoch, Seq, Labels)
        return loss.item(), accx, accy
    
    def step(self, epoch, Seq, Labels):
        n_batch = Seq.size(0)
        n_in = self.n_in
        n_out = self.n_out
        label_dim = self.label_dim
        CE = self.CE
        device = self.device

        Seq = Seq.cuda()
        Labels = torch.argmax(Labels.cuda(), dim=2)
        Labels_in = Labels[:, :n_in]
        Labels_out = Labels[:, n_in:]

        E = self.models[0]
        D = self.models[1]
        labels_in_pred_raw, labels_in_pred, h = E(Seq[:, :n_in])
        
        # loss = CE(labels_in_pred_raw.reshape(n_batch * n_in, label_dim), 
        #           Labels_in.reshape(n_batch * n_in,))
        loss = 0
        accx = calculate_accuracy(Labels_in, labels_in_pred_raw)

        # last_label = labels_in_pred[:, -1, :].unsqueeze(1)
        last_label = torch.zeros((n_batch, 1, label_dim)).cuda()

        Labels_out_pred = []
        prev_label = None
        for t in range(n_out):
            w = (n_out-t)/n_out
            label_target = Labels_out[:, t]
            label_pred_raw, label_pred, h = D(last_label, h)
            loss += w * CE(
                torch.squeeze(label_pred_raw),
                label_target)
            # last_label = label_pred.detach()
            Labels_out_pred.append(label_pred.detach())

            if prev_label is not None:
                # inertia 
                diff = torch.mean((prev_label - label_pred)**2)
                loss += diff

            prev_label = label_pred

        Labels_out_pred = torch.cat(Labels_out_pred, dim=1)
        accy = calculate_accuracy(Labels_out, Labels_out_pred)

        return loss, accx, accy

    def predict(self, Seq, n_out=-1):
        """
        """
        if isinstance(Seq, np.ndarray):
            Seq = torch.from_numpy(Seq)
        n_frames = Seq.size(1)
        E = self.models[0].eval()
        D = self.models[1].eval()
        n_batch = Seq.size(0)
        n_in = self.n_in
        if n_out == -1:
            n_out = self.n_out
        assert n_in == n_frames, Seq.size()
        label_dim = self.label_dim
        with torch.no_grad():
            Seq = Seq.cuda()
            labels_in_pred_raw, labels_in_pred, h = E(Seq)
            last_label = labels_in_pred[:, -1, :].unsqueeze(1)

            Labels_out_pred = []
            for t in range(n_out):
                _, label_pred, h = D(last_label, h)
                last_label = label_pred.detach()
                Labels_out_pred.append(last_label)
            
            Labels_out_pred = torch.cat(Labels_out_pred, dim=1)

            A = labels_in_pred.cpu().numpy()
            B = Labels_out_pred.cpu().numpy()

        # Labels_pred = np.concatenate([A, B], axis=1)
        return B

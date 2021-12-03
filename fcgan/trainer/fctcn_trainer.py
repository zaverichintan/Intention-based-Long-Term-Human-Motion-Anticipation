from junn.training import Trainer
from fcgan.models.fctcn import SegmentForecastingEncoder, SegmentForecastingDecoder
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


class SegmentForecasting_Trainer(Trainer):

    def __init__(self, hidden_units, device,
                 label_dim, txt, stacks, dim, n_in, n_out,
                 model_seed, force_new_training=False):
        """
        """
        self.n_in = n_in
        self.n_out = n_out
        self.stacks = stacks
        self.CE = nn.CrossEntropyLoss()
        encoder = SegmentForecastingEncoder(
            n_in=n_in, n_out=n_out,
            hidden_units=hidden_units, label_dim=label_dim, txt=txt, stacks=stacks,
            model_seed=model_seed, force_new_training=force_new_training, dim=dim
        )
        decoder = SegmentForecastingDecoder(
            n_in=n_in, n_out=n_out,
            hidden_units=hidden_units, label_dim=label_dim, txt=txt, stacks=stacks,
            model_seed=model_seed, force_new_training=force_new_training, dim=dim
        )

        super().__init__(
            [encoder, decoder], put_all_models_in_common_dir=False, 
            save_only_best_model=False, max_epoch=100, device=device,
        )

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
        n_in = self.n_in
        n_out = self.n_out
        CE = self.CE
        stacks = self.stacks
        device = self.device
        
        Seq = Seq.to(device)
        Labels = Labels.to(device)
        Labels = torch.argmax(Labels, dim=2)
        Labels_X = Labels[:, :n_in]
        Labels_Y = Labels[:, n_in:]
        # Labels_X = torch.argmax(Labels[:, :n_in], dim=2)
        # Labels_Y = torch.argmax(Labels[:, n_in:], dim=2)
        encoder = self.models[0]
        decoder = self.models[1]

        Labels_X_hat, Labels_X_raw_hat, h = encoder(Seq[:, :n_in])

        ce_loss = 0
        ce_loss += CE(Labels_X_raw_hat, Labels_X)
        with torch.no_grad():
            acc_X = calculate_accuracy(Labels_X, Labels_X_raw_hat.permute(0, 2, 1))

        weights = np.linspace(1, 0.1, num=n_out)

        prev_label = Labels_X_hat[:, :, -1].unsqueeze(1)

        acc_Y = []
        for t in range(n_out):
            w = weights[t]
            label, label_raw, h = decoder(prev_label, h)
            target_label = Labels_Y[:, t]
            ce_loss += w * CE(label_raw, target_label)
            prev_label = label.detach()
            with torch.no_grad():
                acc_Y.append(calculate_accuracy(target_label.unsqueeze(1), label))
        acc_Y = np.mean(acc_Y)

        return ce_loss, acc_X, acc_Y

    def predict(self, Seq):
        if isinstance(Seq, np.ndarray):
            Seq = torch.from_numpy(Seq)
        Seq = Seq.to(self.device)
        assert Seq.size(1) == self.n_in
        with torch.no_grad():
            model = self.models[0]
            model.eval()
            out = torch.argmax(model(Seq), dim=1)
            # out = torch.argmax(model(Seq)[-1], dim=1)
            return out.cpu().numpy()

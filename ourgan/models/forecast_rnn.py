from junn.scaffolding import Scaffolding
import torch.nn as nn
from os.path import join
import torch


class Discriminator(Scaffolding):

    def __init__(self, n_out, label_dim, hidden_units, txt,
                 model_seed, force_new_training=False):
        """ ctor """
        self.txt = txt
        self.label_dim = label_dim
        self.hidden_units = hidden_units
        self.n_out = n_out
        super().__init__(
            force_new_training=force_new_training,
            model_seed=model_seed)
        self.network = nn.Sequential(
            nn.Linear(n_out * label_dim, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, 1),
            nn.Sigmoid()
        )
    
    def get_unique_directory(self):
        """
        """
        return join(
            'discs_' + str(self.hidden_units) + '_nout' + str(self.n_out),
            self.txt)
    
    def forward(self, labels):
        """
        :prarams labels: [n_batch x n_out x label_dim]
        """
        n_batch = labels.size(0)
        labels = labels.reshape((n_batch, self.n_out * self.label_dim))
        return self.network(labels)


class ForecastRNN(Scaffolding):

    def __init__(self, hidden_units, dim, label_dim, txt, stacks,
                 n_in=25, n_out=12, model_seed=0,
                 force_new_training=False):
        self.n_out = n_out
        self.txt = join(
            'h' + str(hidden_units) + '_d' + str(dim) + '_ld' + str(label_dim) + '_s' + str(stacks) + '_' + txt,
            'nin' + str(n_in) + '_nout' + str(n_out))
        super().__init__(
            force_new_training=force_new_training,
            model_seed=model_seed)
        self.encoder = nn.GRU(
            input_size=dim, hidden_size=hidden_units,
            num_layers=stacks, batch_first=True,
            dropout=0 if stacks == 1 else 0.3,
            bidirectional=False)
        self.decoder = nn.GRU(
            input_size=1, hidden_size=hidden_units,
            num_layers=stacks, batch_first=True,
            dropout=0 if stacks == 1 else 0.3,
            bidirectional=False)
        self.decode_label = nn.Linear(hidden_units, label_dim)
        self.sm = nn.Softmax(dim=2)
    
    def get_unique_directory(self):
        """
        """
        return join('forecastrnn_', self.txt)
    
    def forward(self, poses, n_out=0):
        """ """
        n_batch = poses.size(0)
        if n_out == 0:
            n_out = self.n_out
        _, h = self.encoder(poses)
        fcinput = torch.zeros((n_batch, n_out, 1)).cuda()
        o, _ = self.decoder(fcinput, h)
        labels_raw = self.decode_label(o)
        labels = self.sm(labels_raw)
        return labels_raw, labels

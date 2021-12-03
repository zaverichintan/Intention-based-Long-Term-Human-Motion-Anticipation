from junn.scaffolding import Scaffolding
import torch.nn as nn
from os.path import join
import torch
from fcgan.models.tcn import TemporalConvNet


class SegmentForecastingEncoder(Scaffolding):

    def __init__(self, hidden_units, dim,
                 label_dim, txt, stacks,
                 model_seed, n_in=25, n_out=50,
                 force_new_training=False):
        self.hidden_units = hidden_units
        self.label_dim = label_dim
        self.stacks = stacks
        self.n_out = n_out
        self.txt = join(
            'ld' + str(dim) + '_' + str(label_dim) +\
            '_' + str(hidden_units) + '_' + '_s' + str(stacks) +\
            '_nin' + str(n_in) + '_to_' + str(n_out),
            txt)
        super().__init__(force_new_training=force_new_training,
                         model_seed=model_seed)
        self.encoder = nn.GRU(
            input_size=dim, hidden_size=hidden_units,
            num_layers=stacks, batch_first=True,
            dropout=0 if stacks == 1 else 0.3,
            bidirectional=True)
        self.decode_label = nn.Linear(hidden_units*2, label_dim)
        self.sm = nn.Softmax(dim=2)

    def get_unique_directory(self):
        """
        """
        return join('forecast_enc_', self.txt)
    
    def forward(self, poses):
        """
        :param poses: {batch x n_frames x dim}
        """
        batchsize = poses.size(0)
        out, h = self.encoder(poses)
        
        predicted_labels_raw = self.decode_label(out)
        predicted_labels = self.sm(predicted_labels_raw)

        h = h.view(self.stacks, 2, batchsize, self.hidden_units)

        fwd, bwd = torch.unbind(h, dim=1)
        h = torch.cat([fwd, bwd], dim=2)
        
        return predicted_labels_raw, predicted_labels, h
    

class SegmentForecastingDecoder(Scaffolding):

    def __init__(self, hidden_units, dim,
                 label_dim, txt, stacks,
                 model_seed, n_in=25, n_out=50,
                 force_new_training=False):
        self.hidden_units = hidden_units
        self.label_dim = label_dim
        self.stacks = stacks
        self.n_out = n_out
        self.txt = join(
            'ld' + str(dim) + '_' + str(label_dim) +\
            '_' + str(hidden_units) + '_' + '_s' + str(stacks) +\
            '_nin' + str(n_in) + '_to_' + str(n_out),
            txt)
        super().__init__(force_new_training=force_new_training,
                         model_seed=model_seed)
        self.decoder = nn.GRU(
            input_size=label_dim, hidden_size=hidden_units,
            num_layers=stacks, batch_first=True,
            dropout=0 if stacks == 1 else 0.3,
            bidirectional=False)
        self.decode_label = nn.Linear(hidden_units, label_dim)
        self.sm = nn.Softmax(dim=2)

    def get_unique_directory(self):
        """
        """
        return join('forecast_dec_', self.txt)
    
    def forward(self, label, h):
        """
        :param label: {batch x 1 x dim}
        """
        batchsize = label.size(0)
        out, h = self.decoder(label, h)
        
        predicted_label_raw = self.decode_label(out)
        predicted_label = self.sm(predicted_label_raw)

        return predicted_label_raw, predicted_label, h

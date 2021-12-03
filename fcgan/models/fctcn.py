from junn.scaffolding import Scaffolding
import torch.nn as nn
from os.path import join
import torch
from fcgan.models.tcn import TemporalConvNet



class SegmentForecastingDecoder(Scaffolding):

    def __init__(self, hidden_units, dim,
                 label_dim, txt, stacks,
                 model_seed, n_in=25, n_out=50,
                 force_new_training=False):
        self.stacks = stacks
        self.hidden_units = hidden_units
        self.label_dim = label_dim
        self.n_out = n_out
        self.txt = join(
            'ld' + str(dim) + '_' + str(label_dim) +\
            '_' + str(hidden_units) + '_' + str(stacks) +\
            '_nin' + str(n_in) + '_to_' + str(n_out), 
            txt)
        super().__init__(force_new_training=force_new_training,
                         model_seed=model_seed)
        self.decoder = nn.GRU(
            input_size=label_dim, hidden_size=hidden_units,
            num_layers=1, batch_first=True
        )
        self.decode_label = nn.Linear(hidden_units, label_dim)
        self.sm = nn.Softmax(dim=2)

    def get_unique_directory(self):
        """
        """
        return join('seg_fc_decoder_', self.txt)
    
    def forward(self, prev_label, h):
        n_batch = prev_label.size(0)
        out, h = self.decoder(prev_label)
        label_raw = self.decode_label(out)
        label = self.sm(label_raw).detach()
        label_raw = label_raw.reshape(n_batch, self.label_dim)
        return label, label_raw, h



class SegmentForecastingEncoder(Scaffolding):

    def __init__(self, hidden_units, dim,
                 label_dim, txt, stacks,
                 model_seed, n_in=25, n_out=50,
                 force_new_training=False):
        self.stacks = stacks
        self.hidden_units = hidden_units
        self.n_out = n_out
        self.txt = join(
            'ld' + str(dim) + '_' + str(label_dim) +\
            '_' + str(hidden_units) + '_' + str(stacks) +\
            '_nin' + str(n_in) + '_to_' + str(n_out), 
            txt)
        super().__init__(force_new_training=force_new_training,
                         model_seed=model_seed)

        self.encoder_tcn = TemporalConvNet(
            num_inputs=dim, 
            num_channels=[hidden_units, hidden_units, hidden_units, hidden_units])
        
        self.decode_inputposes = TemporalConvNet(
            num_inputs=dim+hidden_units,
            num_channels=[hidden_units, hidden_units, label_dim]
        )

        self.encoder = nn.GRU(
            input_size=hidden_units, hidden_size=hidden_units,
            num_layers=1, batch_first=True
        )
        self.sm = nn.Softmax(dim=1)
        

    def get_unique_directory(self):
        """
        """
        return join('seg_fc_encoder', self.txt)
    
    def forward(self, poses):
        n_batch = poses.size(0)
        poses = poses.permute(0, 2, 1)  # for cnn

        h_in = self.encoder_tcn(poses)

        poses_with_h = torch.cat([h_in, poses], dim=1)
        labels_raw_in_hat = self.decode_inputposes(poses_with_h)
        labels_in_hat = self.sm(labels_raw_in_hat).detach()

        h_in = h_in.permute(0, 2, 1)

        _, h = self.encoder(h_in)

        return labels_in_hat, labels_raw_in_hat, h

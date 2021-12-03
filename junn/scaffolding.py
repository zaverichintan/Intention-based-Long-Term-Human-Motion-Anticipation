from abc import abstractmethod
import time
from os.path import isdir, isfile, join, isfile, dirname
from os import makedirs, listdir
import shutil
import numpy as np
import torch
import torch.nn as nn
from junn.settings import get_data_loc
import pandas as pd
from tqdm.auto import tqdm


class Scaffolding(nn.Module):

    def __init__(self, force_new_training, model_seed=0, name='', verbose=True):
        """
        :param force_new_training:
        :param model_seed: identifiy different training runs for the same model
        """
        super(Scaffolding, self).__init__()
        self.name = name
        self.project_folder = ''
        self.model_seed = model_seed
        self.is_weights_loaded = False
        self.force_new_training = force_new_training
        self.scaffolding_is_init = False
        self.verbose = verbose
    
    def init(self):
        if not self.scaffolding_is_init:
            force_new_training = self.force_new_training
            train_dir = self.get_train_dir()
            if isdir(train_dir) and force_new_training:
                print('[hma-with-symbolic-label][model] - delete dir:', train_dir)
                shutil.rmtree(train_dir)
                time.sleep(.5)
            if not isdir(train_dir):
                makedirs(train_dir)
            fweights = self.get_weights_file()
            if isfile(fweights):
                self.is_weights_loaded = True
            self.scaffolding_is_init = True

    @abstractmethod
    def get_unique_directory(self):
        raise NotImplementedError

    def number_of_parameters(self):
        total_sum = []
        for param in self.parameters():
            total_sum.append(np.product(param.size()))
        return np.sum(total_sum)

    def prettyprint_number_of_parameters(self):
        n_params = self.number_of_parameters()
        return '{:,}'.format(n_params)

    def load_weights_if_possible(self):
        if isfile(self.get_weights_file(name=self.name)):
            checkpoint = torch.load(self.get_weights_file(name=self.name))
            self.load_state_dict(checkpoint['model_state_dict'])
            self.is_weights_loaded = True
            return True
        self.is_weights_loaded = False
        return False
    
    def load_weights_for_epoch(self, epoch):
        fname = 'weights_ep%04d.h5' % epoch
        self.load_specific_weights(fname)
    
    def load_specific_weights(self, fname):
        """
        """
        if not isfile(fname):
            loc = dirname(self.get_weights_file())
            fname = join(loc, fname)
        assert isfile(fname), fname
        checkpoint = torch.load(fname)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.is_weights_loaded = True
        if self.verbose:
            print('LOADED: ', fname)
    
    def count_trained_epochs(self):
        fname = self.get_log_file()
        if isfile(fname):
            ff = pd.read_csv(fname)
            return max(ff['epoch'])
        else:
            return 0

    def find_best_weight(self, callback_fn, return_all_scores=False):
        """
        :param callback_fn: {function} def callback_fn(model): {return score}
        """
        is_verbose_bkp = self.verbose
        self.verbose = False
        weight_files = self.list_all_weights_in_training_dir()
        assert len(weight_files) > 0, "There are no weight files!"

        scores = []
        for wfile in tqdm(weight_files):
            self.load_specific_weights(wfile)
            score = callback_fn(self)
            scores.append(score)
        best_score = np.argmin(scores)
        
        self.verbose = is_verbose_bkp  # reset verbose to old state

        if return_all_scores:
            return scores, weight_files
        else:
            return weight_files[best_score]

    def list_all_weights_in_training_dir(self):
        """ lists all the weights file that were written
        """
        loc = dirname(self.get_weights_file())
        return [f for f in sorted(listdir(loc)) if f.startswith('weight') and f.endswith('.h5')]
    
    def save_weights(self, epoch=-1, optim=None, name=''):
        if optim is None:
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.state_dict()
            }, self.get_weights_file(name=name))
        else:
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.state_dict(),
                'optim_state_dict': optim.state_dict()
            }, self.get_weights_file(name=name))

    def get_train_dir(self):
        
        if len(self.project_folder) > 0:
            train_dir = self.project_folder
        else:
            train_dir = join(get_data_loc(), 'training')
        return join(join(train_dir, self.get_unique_directory()),
                    'seed' + str(self.model_seed))

    def get_weights_file(self, name=''):
        return join(self.get_train_dir(), 'weights' + name + '.h5')

    def get_log_file(self):
        return join(self.get_train_dir(), 'training.csv')

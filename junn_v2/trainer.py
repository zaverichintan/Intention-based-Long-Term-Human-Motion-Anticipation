import torch
import junn.utils as utils
import shutil
import json

from abc import abstractmethod
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from os.path import isfile, isdir, join
from os import makedirs
import numpy as np
import pandas as pd
import junn.console as console

from junn.settings import get_data_loc
from typing import Dict


class AbstractTrainer:
    def __init__(
        self,
        models: Dict,
        train_params: Dict,
        device,
        force_new_training=False,
        verbose=True,
        model_seed=0,
        max_epoch=999,
        epoch=-1,
        project_folder: str = "",
    ):
        """
        models: {"model_name" : ({nn.Module}, {nn.Optimizer})}
        train_params: define the entire training, e.g. param count etc
        """
        assert isinstance(models, dict)
        assert isinstance(train_params, dict)
        torch.manual_seed(model_seed)
        np.random.seed(model_seed)
        self.project_folder = project_folder
        self.unique_name = utils.uniqueify(train_params)
        self.max_epoch = max_epoch
        self.verbose = verbose
        self.device = device
        self.models = models

        # ~~~~~~~~~~~~~~~~~~
        # --- initialize ---
        # ~~~~~~~~~~~~~~~~~~
        train_dir = self.get_train_dir()
        if force_new_training and isdir(train_dir):
            console.warning(f"deleting {train_dir}")
            shutil.rmtree(train_dir)

        if not isdir(train_dir):
            if verbose:
                console.warning(f"create {train_dir}")
            makedirs(train_dir)
        elif verbose:
            console.info(f"found train dir {train_dir}")
            
        # -- save parameters --
        fname_params = join(train_dir, "train_params.json")
        if not isfile(fname_params):
            with open(fname_params, "w") as f:
                json.dump(train_params, f)

        # -- tensorboard --
        self.writer = SummaryWriter(log_dir=join(self.get_train_dir(), "tfboard"))

        # -- load log file --
        fname_log = self.get_log_file()
        if isfile(fname_log):
            data = pd.read_csv(self.get_log_file())
            current_epoch = max(data["epoch"].values) + 1
            self.current_epoch = current_epoch
            console.info(f"load log file and epoch {current_epoch}")
            if epoch > -1:
                assert epoch <= current_epoch, f"{epoch} > {current_epoch}"

            # load weights
            utils.load_models(models, train_dir, epoch)
            console.success("all model weights are loaded!")

        else:
            self.current_epoch = 0
            console.warning("found empty training folder")

        for net, _ in models.values():
            net.to(device)

    @abstractmethod
    def trainer_name(self):
        raise NotImplementedError

    @abstractmethod
    def train_step(self, epoch, Data, models):
        raise NotImplementedError

    @abstractmethod
    def val_step(self, epoch, Data, models):
        raise NotImplementedError

    @abstractmethod
    def on_epoch_end(self, epoch):
        raise NotImplementedError

    def loss_names(self):
        """defines the names of the losses.
        Must coincide with the number of returned
        losses from train_step/val_step
        """
        return ["loss"]

    def prettyprint_number_of_parameters(self):
        """for param numbers"""
        for name in sorted(self.models.keys()):
            n_params = utils.prettyprint_number_of_parameters(self.models[name][0])
            console.info(f"{name}: {n_params}")

    def get_train_dir(self):
        """gets the training dir"""
        if len(self.project_folder) > 0:
            train_dir = self.project_folder
        else:
            train_dir = get_data_loc()
        train_dir = join(train_dir, "training")
        return join(join(train_dir, self.trainer_name()), self.unique_name)

    def get_log_file(self):
        return join(self.get_train_dir(), "training.csv")

    def format_loss(self, losses):
        if not isinstance(losses, list):
            losses = [losses]
        names = self.loss_names()
        assert len(names) == len(losses), (
            "incompatible names:" + str(len(names)) + " vs " + str(len(losses))
        )
        txt = ""
        for loss, name in zip(losses, names):
            txt += name + ":{:.4f} ".format(loss)
        return txt

    def store_losses_to_file(self, epoch, train_losses, val_losses):
        if not isinstance(train_losses, list):
            train_losses = [train_losses]
        if not isinstance(val_losses, list):
            val_losses = [val_losses]
        names = self.loss_names()
        assert len(names) == len(train_losses)
        assert len(names) == len(val_losses)

        data_entry = {"epoch": [epoch]}

        for loss, name in zip(train_losses, names):
            keyname = "train_" + name
            data_entry[keyname] = [loss]
            self.writer.add_scalar(keyname, loss, epoch)
        for loss, name in zip(val_losses, names):
            keyname = "val_" + name
            data_entry[keyname] = [loss]
            self.writer.add_scalar(keyname, loss, epoch)

        df = pd.DataFrame(data_entry)

        if isfile(self.get_log_file()):
            with open(self.get_log_file(), "a") as f:
                df.to_csv(f, header=False)
        else:
            with open(self.get_log_file(), "w") as f:
                df.to_csv(f, header=True)

    def run(self, dl_train, dl_val):
        """
        :param dl_train: {pytorch dataloader}
        :param dl_val: {pytorch dataloader}
        """
        start_epoch = self.current_epoch
        end_epoch = self.max_epoch
        models = self.models

        for epoch in range(start_epoch, end_epoch):

            Train_losses = {}
            Val_losses = {}
            for name in self.loss_names():
                Train_losses[name] = []
                Val_losses[name] = []

            if self.verbose:
                console.info(f"epoch {epoch} -> " + self.get_train_dir())

            # ~~ TRAIN STEP ~~
            for net, _ in models.values():
                net.train()
            train_tqdm = tqdm(dl_train)
            for Data in train_tqdm:
                train_losses = self.train_step(epoch, Data, models)
                if not isinstance(train_losses, list) and not isinstance(
                    train_losses, tuple
                ):
                    train_losses = [train_losses]
                for i, name in enumerate(self.loss_names()):
                    Train_losses[name].append(train_losses[i])

                train_losses = []
                for name in self.loss_names():
                    train_losses.append(np.mean(Train_losses[name]))

                train_tqdm.set_description(
                    "[train] " + self.format_loss(train_losses)
                )

            # ~~ EVAL STEP ~~
            if dl_val is not None:
                for net, _ in models.values():
                    net.eval()

                with torch.no_grad():
                    val_tqdm = tqdm(dl_val)
                    for Data in val_tqdm:
                        val_losses = self.val_step(epoch, Data, models)

                        for i, name in enumerate(self.loss_names()):
                            Val_losses[name].append(val_losses[i])

                        val_losses = []
                        for name in self.loss_names():
                            val_losses.append(np.mean(Val_losses[name]))

                        val_tqdm.set_description(
                            "[val] " + self.format_loss(val_losses)
                        )

            # -- save epoch
            val_losses = []
            train_losses = []
            for name in self.loss_names():
                if dl_val is None:
                    val_losses.append(np.mean(Train_losses[name]))
                else:
                    val_losses.append(np.mean(Val_losses[name]))
                train_losses.append(np.mean(Train_losses[name]))

            self.store_losses_to_file(epoch, train_losses, val_losses)

            try:
                self.on_epoch_end(epoch)
            except NotImplementedError:
                pass

            utils.save_models(models, epoch, self.get_train_dir(), save_for_epoch=True)

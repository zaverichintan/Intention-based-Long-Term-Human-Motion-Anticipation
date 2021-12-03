import numpy as np
import torch
import hashlib

from os.path import isdir, isfile, join
from os import makedirs
from typing import Dict


def number_of_parameters(model):
    """calcualate the number of model params"""
    total_sum = []
    for param in model.parameters():
        total_sum.append(np.product(param.size()))
    return np.sum(total_sum)


def prettyprint_number_of_parameters(model):
    """calcualate the number of model params"""
    n_params = number_of_parameters(model)
    return "{:,}".format(n_params)


def load_models(models: Dict, training_path: str, epoch: int = -1):
    """load the models from file"""
    if epoch != -1:
        training_path = join(training_path, "per_epoch")
    assert isdir(training_path), training_path
    for name in sorted(models.keys()):
        net, opt = models[name]
        if epoch == -1:
            fname = join(training_path, name + ".pth")
        else:
            fname = join(training_path, name + "_%05d.pth" % epoch)
        assert isfile(fname), fname
        checkpoint = torch.load(fname)
        net.load_state_dict(checkpoint["model_state_dict"])
        opt.load_state_dict(checkpoint["optim_state_dict"])


def save_models(models: Dict, epoch: int, training_path: str, save_for_epoch=True):
    """
    models: {"model_name": (nn.Module, nn.Optimizer)}
    """
    assert isdir(training_path), training_path
    for name in models.keys():
        net, opt = models[name]
        fname = join(training_path, name + ".pth")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": net.state_dict(),
                "optim_state_dict": opt.state_dict(),
            },
            fname,
        )
        if save_for_epoch:
            training_path_ep = join(training_path, "per_epoch")
            if not isdir(training_path_ep):
                makedirs(training_path_ep)
            fname = join(training_path_ep, name + "_%05d.pth" % epoch)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": net.state_dict(),
                    "optim_state_dict": opt.state_dict(),
                },
                fname,
            )


def uniqueify(params: Dict):
    """
    loop over all parameters and create a unique hash str. This way we
    can ensure that different model parameterizations get different
    directories
    """
    txt = ""
    for key in sorted(params.keys()):
        param = params[key]
        subtxt = f"{key}--->"
        if isinstance(param, list):
            for p in sorted(param):
                subtxt += str(p)
        else:
            subtxt += str(param)
        txt += subtxt
    pseudo_hash = int(hashlib.sha256(txt.encode("utf-8")).hexdigest(), 16) % 10 ** 12
    return str(pseudo_hash)

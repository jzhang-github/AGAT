# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 22:01:12 2023

@author: ZHANG Jun
"""

import os
import json
import copy

import torch

def save_model(model, model_save_dir='agat_model'):
    """Saving PyTorch model to the disk. Save PyTorch model, including parameters and structure. See: https://pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html

    :param model: A PyTorch-based model.
    :type model: PyTorch-based model.
    :param model_save_dir: A directory to store the model, defaults to 'agat_model'
    :type model_save_dir: str, optional
    :output: A file saved to the disk under ``model_save_dir``.
    :outputtype: A file.

    """
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    torch.save(model, os.path.join(model_save_dir, 'agat.pth'))

def load_model(model_save_dir='agat_model', device='cuda'):
    """Loading PyTorch model from the disk.

    :param model_save_dir: A directory to store the model, defaults to 'agat_model'
    :type model_save_dir: str, optional
    :param device: Device for the loaded model, defaults to 'cuda'
    :type device: str, optional
    :return: A PyTorch-based model.
    :rtype: PyTorch-based model.

    """
    device = torch.device(device)
    if device.type == 'cuda':
        new_model = torch.load(os.path.join(model_save_dir, 'agat.pth'))
    elif device.type == 'cpu':
        new_model = torch.load(os.path.join(model_save_dir, 'agat.pth'),
                               map_location=torch.device(device))
    new_model.eval()
    new_model = new_model.to(device)
    new_model.device = device
    return new_model

def save_state_dict(model, state_dict_save_dir='agat_model', **kwargs):
    """Saving state dict (model weigths and other input info) to the disk. See: https://pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html

    :param model: A PyTorch-based model.
    :type model: PyTorch-based model.
    :param state_dict_save_dir: A directory to store the model state dict (model weigths and other input info), defaults to 'agat_model'
    :type state_dict_save_dir: str, optional
    :param **kwargs: More information you want to save.
    :type **kwargs: kwargs
    :output: A file saved to the disk under ``model_save_dir``.
    :outputtype: A file

    """
    if not os.path.exists(state_dict_save_dir):
        os.makedirs(state_dict_save_dir)
    checkpoint_dict = {**{'model_state_dict': model.state_dict()}, **kwargs}
    torch.save(checkpoint_dict, os.path.join(state_dict_save_dir, 'agat_state_dict.pth'))

def load_state_dict(state_dict_save_dir='agat_model'):
    """Loading state dict (model weigths and other info) from the disk. See: https://pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html

    :param state_dict_save_dir:  A directory to store the model state dict (model weigths and other info), defaults to 'agat_model'
    :type state_dict_save_dir: str, optional
    :return: State dict.
    :rtype: TYPE

    .. note::
        Reconstruct a model/optimizer before using the loaded state dict.

        Example::

            model = PotentialModel(...)
            model.load_state_dict(checkpoint['model_state_dict'])
            new_model.eval()
            model = model.to(device)
            model.device = device
            optimizer = ...
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    """

    checkpoint_dict = torch.load(os.path.join(state_dict_save_dir, 'agat_state_dict.pth'))
    return checkpoint_dict

def config_parser(config):
    """Parse the input configurations/settings.

    :param config: configurations
    :type config: str/dict. if str, load from the json file.
    :raises TypeError: DESCRIPTION
    :return: TypeError('Wrong configuration type.')
    :rtype: TypeError

    """

    if isinstance(config, dict):
        return config
    elif isinstance(config, str):
        with open(config, 'r') as config_f:
            return json.load(config_f)
    elif isinstance(config, type(None)):
        return {}
    else:
        raise TypeError('Wrong configuration type.')

class EarlyStopping:
    def __init__(self, model, logger, patience=10, model_save_dir='model_save_dir'):
        """Stop training when model performance stop improving after some steps.

        :param model: AGAT model
        :type model: torch.nn
        :param logger: I/O file
        :type logger: _io.TextIOWrapper
        :param patience: Stop patience, defaults to 10
        :type patience: int, optional
        :param model_save_dir: A directory to save the model, defaults to 'model_save_dir'
        :type model_save_dir: str, optional


        """

        self.model      = model
        self.patience   = patience
        self.counter    = 0
        self.best_score = None
        self.update     = None
        self.early_stop = False
        self.logger     = logger
        self.model_save_dir = model_save_dir

        if not os.path.exists(self.model_save_dir):
            os.mkdir(model_save_dir)
        self.save_model_info()

    def step(self, score, epoch, model, optimizer):
        if self.best_score is None:
            self.best_score = score
            self.update = True
            # self.save_model(model, model_save_dir=self.model_save_dir)
            # self.save_checkpoint(model, model_save_dir=self.model_save_dir)
        elif score > self.best_score:
            self.update = False
            self.counter += 1
            print(f'User log: EarlyStopping counter: {self.counter} out of {self.patience}',
                  file=self.logger)
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.update     = True
            self.save_model(model)
            save_state_dict(model, state_dict_save_dir=self.model_save_dir,
                            optimizer_state_dict=optimizer.state_dict(),
                            epoch=epoch, total_loss=score)
            self.counter = 0
        return self.early_stop

    def save_model(self, model):
        '''Saves model when validation loss decrease.'''
        torch.save(model, os.path.join(self.model_save_dir, 'agat.pth'))
        print(f'User info: Save model with the best score: {self.best_score}',
              file=self.logger)

    # def save_checkpoint(self, model):
    #     '''Saves model when validation loss decrease.'''
    #     torch.save({model.state_dict()}, os.path.join(self.model_save_dir, 'agat_state_dict.pth'))

    def save_model_info(self):
        info = copy.deepcopy(self.model.__dict__)
        info = {k:v for k,v in info.items() if isinstance(v, (str, list, int, float))}
        with open(os.path.join(self.model_save_dir, 'agat_model.json'), 'w') as f:
            json.dump(info, f, indent=4)

def load_graph_build_method(path):
    """ Load graph building scheme. This file is normally saved when you build your dataset.

    :param path: Directory for storing ``graph_build_scheme.json`` file.
    :type path: str
    :return: A dict denotes how to build the graph.
    :rtype: dict

    """

    json_file  = path

    assert os.path.exists(json_file), f"{json_file} file dose not exist."
    with open(json_file, 'r') as jsonf:
        graph_build_scheme = json.load(jsonf)
    return graph_build_scheme

def PearsonR(y_true, y_pred):
    """Calculating the Pearson coefficient.

    :param y_true: The first torch.tensor.
    :type y_true: torch.Tensor
    :param y_pred: The second torch.tensor.
    :type y_pred: torch.Tensor
    :return: Pearson coefficient
    :rtype: torch.Tensor

    .. Note::

        It looks like the `torch.jit.script` decorator is not helping in comuputing large `torch.tensor`, see `agat/test/tesor_computation_test.py` for more details.

    """
    ave_y_true = torch.mean(y_true)
    ave_y_pred = torch.mean(y_pred)

    y_true_diff = y_true - ave_y_true
    y_pred_diff = y_pred - ave_y_pred

    above = torch.sum(torch.mul(y_true_diff, y_pred_diff))
    below = torch.mul(torch.sqrt(torch.sum(torch.square(y_true_diff))),
                             torch.sqrt(torch.sum(torch.square(y_pred_diff))))
    return torch.divide(above, below)

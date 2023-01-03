#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Author'
__email__ = 'Email'


# dependency
# built-in
import time, copy, random, pickle
# public
import transformers
import numpy as np
import torch
# private
from src.models import tfm
from src.trainers import seq2seq as seq2seq_trainer
from src.testers import seq2seq as seq2seq_tester
from src.datasets import quora


def save_pickle(path, obj):
    """
    To save a object as a pickle file.
    """
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_pickle(path):
    """
    To load object from pickle file.
    """
    with open(path, 'rb') as f:
        return pickle.load(f)

def show_sample(xs: list, ys: list, if_return: bool=False) -> None:
    """
    Randomly show, (and) return sample pairs
    """
    idx = random.randrange(0, len(xs))
    print(xs[idx])
    print(ys[idx])
    if if_return:
        return xs[idx], ys[idx]

def show_rand_sample(srcs, tars, preds): 
    src, tar, pred = random.choice([(src, tar, pred) for src, tar, pred in zip(srcs, tars, preds)])
    return '\nsrc: {}\ntar: {}\npred: {}'.format(src, tar, pred)

def set_seed(seed):
    # random
    random.seed(seed)
    # numpy
    np.random.seed(seed)
    # torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # transformers
    transformers.set_seed(seed)
    # cuda
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_model(config):
    if config.model == 'tfm':
        return tfm.ModelGraph(config)
    raise NotImplementedError

def get_trainer(config):
    return seq2seq_trainer.Trainer(config)

def get_tester(trainer):
    return seq2seq_tester.Tester(trainer)

def get_dataset(config):
    match config.task:
        case 'ori_quora':
            return quora.Dataset
        case 'sep_quora':
            return quora.Dataset
    raise NotImplementedError

def get_optim_params(model, config):
    no_decay = ['bias', 'LayerNorm.weight']
    parameters = []
    parameters.append(
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 
         'weight_decay': config.weight_decay})
    parameters.append(
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0})
    return parameters

def process_masks(masks, config, training, do_copy=True):
    """
    for training:
        + 80% of the time: keep mask
        + 10% of the time: replace with a random mask
        + 10% of the time: replace with an inference mask
    for inference:
        use mask inference token id only
    """
    if do_copy:
        masks = copy.deepcopy(masks)
    if training:
        new_masks = []
        for m in masks:
            rand = np.random.choice(range(3), p=config.mask_weights)
            match rand:
                case 0:  # keep the label mask
                    pass
                case 1:  # replace with a random mask
                    low = 0
                    high = config.mask_pad_token_id  # 3
                    size = m.shape
                    # in range of [0, 2]
                    m = torch.randint(low=low, high=high, size=size)
                case 2:  # replace with an inference mask
                    m = torch.full(m.shape, config.mask_infer_token_id)
                case other:
                    raise NotImplementedError
            new_masks.append(m)
    else:
        # inference mask only for validation and test
        new_masks = [torch.full(m.shape, config.mask_infer_token_id) for m in masks]
    return new_masks

def pad_masks(xs, masks, config):
    pad_masks = torch.full(xs.shape, config.mask_pad_token_id)
    for i in range(pad_masks.shape[0]):
        pad_idx = (xs[i] != config.pad_token_id).sum().item()
        pad_masks[i][:pad_idx] = masks[i][:pad_idx]
        # the first token is always 0 to delete bos
        pad_masks[i][0] = config.mask_delete_token_id
    return pad_masks

def unify_white_space(x: str) -> str:
    return ' '.join(x.split())
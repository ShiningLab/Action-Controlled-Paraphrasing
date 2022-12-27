#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Author'
__email__ = 'Email'


# dependency
# built-in

# public
import numpy as np
from torch.utils.data import Dataset
# private
from ..utils import helper


class Dataset(Dataset):
    """docstring for Dataset"""
    def __init__(self, mode, tokenizer, config, augmenator=None):
        """
            mode (str): train, val, or test
        """
        super(Dataset, self).__init__()
        self.mode = mode
        self.tokenizer = tokenizer
        self.config = config
        data_dict = helper.load_pickle(config.DATA_PKL)
        self.raw_xs = data_dict[mode]['xs']
        self.raw_ys = data_dict[mode]['ys']
        if augmenator:
            self.raw_xs, self.raw_ys = augmenator.do_aug(self.raw_xs, self.raw_ys)
        self.data_size = len(self.raw_xs)
        self.preprocess()

    def __len__(self): 
        return self.data_size

    def __getitem__(self, idx):
        raw_x, raw_y = self.raw_xs[idx], self.raw_ys[idx]
        x = self.xs_dict.input_ids[idx]
        y = self.ys_dict.input_ids[idx]
        # for mask control
        if self.config.mask:
            mask = self.masks[idx]
            return raw_x, raw_y, x, y, mask
        else:
            return raw_x, raw_y, x, y

    def preprocess(self):
        # dict_keys(['input_ids', 'token_type_ids', 'attention_mask'])
        self.xs_dict = self.tokenizer.batch_encode_plus(
            self.raw_xs
            , add_special_tokens=True
            , return_tensors='pt'
            , padding=True
         )
        self.en_max_len = max(len(_) for _ in self.xs_dict.input_ids)
        self.ys_dict = self.tokenizer.batch_encode_plus(
            self.raw_ys
            , add_special_tokens=True
            , return_tensors='pt'
            , padding=True
         )
        self.de_max_len = max(len(_) for _ in self.ys_dict.input_ids)
        # for mask control
        if self.config.mask:
            self.masks = []
            for x, y in zip(self.xs_dict.input_ids, self.ys_dict.input_ids):
                shared_tokens = set(np.unique(x.numpy())) & set(np.unique(y.numpy()))
                shared_tokens.discard(self.config.pad_token_id)
                mask = sum([x == t for t in shared_tokens])
                self.masks.append(mask)
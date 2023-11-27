#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Author'
__email__ = 'Email'


# dependency
# built-in
# public
from torch.utils import data as torch_data
# private
from src import helper


class PGDataset(torch_data.Dataset):
    """Paraphrase Generation Datasets"""
    def __init__(self, mode, config, tokenizer, sample_size=None):
        super(PGDataset, self).__init__()
        assert mode in ['train', 'val', 'test']
        self.mode = mode
        self.config = config
        self.tokenizer = tokenizer
        self.sample_size = sample_size
        self.get_data()

    def __len__(self): 
        return self.data_size

    def get_data(self):
        data_dict = helper.load_pickle(self.config.DATA_PKL)
        self.raw_xs_list = data_dict[self.mode]['xs']
        self.raw_ys_list = data_dict[self.mode]['ys']
        self.data_size = len(self.raw_xs_list)
        if self.sample_size:
            self.data_size = self.sample_size

    def get_instance(self, idx):
        return self.raw_xs_list[idx], self.raw_ys_list[idx]

    def collate_fn(self, batch):
        raw_xs, raw_ys = list(map(list, zip(*batch)))
        # dict_keys(['input_ids', 'token_type_ids', 'attention_mask'])
        # encode source texts
        xs = self.tokenizer.batch_encode_plus(
            raw_xs
            , add_special_tokens=True
            , return_tensors='pt'
            , padding=True
        )
        if self.mode == 'train':
            # encode target texts
            ys = self.tokenizer.batch_encode_plus(
                raw_ys
                , add_special_tokens=True
                , return_tensors='pt'
                , padding=True
            ).input_ids[:, 1:].clone()
        # return
        match self.mode:
            case 'train':
                return xs, ys
            case 'val':
                return xs, raw_xs, raw_ys
            case 'test':
                return xs, raw_xs, raw_ys
            case _:
                raise NotImplementedError

    def __getitem__(self, idx):
        return self.get_instance(idx)
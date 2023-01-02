#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Author'
__email__ = 'Email'


# dependency
# built-in
import os, copy
# public
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer
from bert_score import BERTScorer
from gramformer import Gramformer
# private
from src.augment import utils
from src.augment.base import Base_Aug


class XXCopy(Base_Aug):
    """docstring for XXCopy"""
    def __init__(self, config, **kwargs):
        super(XXCopy, self).__init__(config)
        self.update_config(**kwargs)

    def update_config(self, **kwargs):
        # update configuration accordingly
        for k,v in kwargs.items():
            setattr(self.config, k, v)

    def do_aug(self, raw_xs_list, raw_ys_list):
        # x to x
        xs_list, ys_list = raw_xs_list + raw_xs_list, raw_ys_list + raw_xs_list
        # y to y
        xs_list, ys_list = xs_list + ys_list, ys_list + ys_list
        # remove duplicates
        xs_list, ys_list = utils.remove_duplicates(xs_list, ys_list)
        return xs_list, ys_list


class YXSwitch(Base_Aug):
    """docstring for YXSwitch"""
    def __init__(self, config, **kwargs):
        super(YXSwitch, self).__init__(config)
        self.update_config(**kwargs)

    def update_config(self, **kwargs):
        # update configuration accordingly
        for k,v in kwargs.items():
            setattr(self.config, k, v)

    def do_aug(self, raw_xs_list, raw_ys_list):
        # x to y + y to x
        xs_list = raw_xs_list + raw_ys_list
        ys_list = raw_ys_list + raw_xs_list
        # remove duplicates
        xs_list, ys_list = utils.remove_duplicates(xs_list, ys_list)
        return xs_list, ys_list


class LinearDecompose(Base_Aug):
    """docstring for LinearDecompose"""
    def __init__(self, config, **kwargs):
        super(LinearDecompose, self).__init__(config)
        self.update_config(**kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.ENCODER_PATH)

    def update_config(self, **kwargs):
        # update configuration accordingly
        for k,v in kwargs.items():
            setattr(self.config, k, v)

    def do_aug(self, raw_xs_list, raw_ys_list):
        # linear decompositional augmentation
        aug_xs_list, aug_ys_list = [], []
        for x, y in zip(tqdm(raw_xs_list), raw_ys_list):
            aug_x, aug_y = utils.linear_decompose(x, y, self.tokenizer)
            if aug_x and aug_y and aug_x != aug_y:
                aug_xs_list.append(aug_x)
                aug_ys_list.append(aug_y)
        xs_list = raw_xs_list + aug_xs_list
        ys_list = raw_ys_list + aug_ys_list
        # remove duplicates
        xs_list, ys_list = utils.remove_duplicates(xs_list, ys_list)
        return xs_list, ys_list


class CustomDataset(Dataset):
    """docstring for CustomDataset"""
    def __init__(self, xs, ys, aug_xs_list):
        super(CustomDataset, self).__init__()
        self.xs = xs
        self.ys = ys
        self.aug_xs_list = aug_xs_list
        self.data_size = len(self.xs)
    def __len__(self): 
        return self.data_size

    def __getitem__(self, idx):
        return self.xs[idx], self.ys[idx], self.aug_xs_list[idx]


class LinearCompose(Base_Aug):
    """docstring for LinearCompose"""
    def __init__(self, config, **kwargs):
        super(LinearCompose, self).__init__(config)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.ENCODER_PATH)
        self.scorer = BERTScorer(
            lang="en"
            , model_type=self.config.SCORER_PATH
            # https://github.com/Tiiiger/bert_score/blob/master/bert_score/utils.py#L119
            , num_layers=18
            , device=self.config.device
            , batch_size=self.config.eval_batch_size
            , nthreads=self.config.num_workers
            , rescale_with_baseline=True
            , baseline_path=os.path.join(self.config.SCORER_PATH, f'{self.config.scorer}.tsv')
            )
        self.corrector = Gramformer(
            models=1
            , use_gpu=config.device == 'cuda'
            )

    def update_config(self, **kwargs):
        # update configuration accordingly
        for k,v in kwargs.items():
            setattr(self.config, k, v)

    def preprocess(self, x: str) -> str:
        x = self.tokenizer.encode(
            x
            , add_special_tokens=False
            , truncation=True
            , max_length=self.config.en_max_len
            )
        x = self.tokenizer.decode(
            x
            , skip_special_tokens=True
            )
        return x

    def do_aug(self, raw_xs_list, raw_ys_list):
        # linear decompose
        compo_xs_list, compo_ys_list = [], []
        for x, y in zip(tqdm(raw_xs_list), raw_ys_list):
            # turn low from 1 to 2 for a better decomposition
            compo_x, compo_y = utils.linear_decompose(
                x, y, self.tokenizer, low=self.config.lc_low)
            if compo_x and compo_y and compo_x != compo_y:
                compo_xs_list.append(compo_x)
                compo_ys_list.append(compo_y)
        # sort from long to short
        compo_xs_list, compo_ys_list = map(
            list, zip(*sorted(zip(compo_xs_list, compo_ys_list)
                , key=lambda x: len(x[0]), reverse=True)))
        # linear compose
        aug_xs_list = []
        for x, y in zip(tqdm(raw_xs_list), raw_ys_list):
            aug_xs = []
            for c_x, c_y in zip(compo_xs_list, compo_ys_list):
                aug_x = None
                if x.startswith(c_x):
                    aug_x = c_y + x[len(c_x):]
                elif x.endswith(c_x):
                    aug_x = x[:len(x)-len(c_x)] + c_y
                if aug_x:
                    # aug_x = self.corrector.correct(aug_x, max_candidates=1).pop()
                    if aug_x != x and aug_x != y and aug_x not in aug_xs:
                        aug_xs.append(aug_x)
                        if len(aug_xs) >= self.config.lc_compo_size:
                            break
            aug_xs_list.append(aug_xs)
        # initialize dataset and dataloader
        dataset = CustomDataset(raw_xs_list, raw_ys_list, aug_xs_list)
        dataloader = DataLoader(
            dataset
            , batch_size=1
            , shuffle=False
            , num_workers=self.config.num_workers
            , pin_memory=self.config.pin_memory
            , drop_last=False
            )
        # filer samples via corrector and scorer
        aug_xs_list, aug_ys_list = [], []
        for x, y, aug_xs in tqdm(dataloader):
            base_f1 = self.scorer.score(x, y)[-1].item()
            if aug_xs:
                # list of list of string
                xs_f1 = self.scorer.score(x*len(aug_xs), aug_xs)[-1]
                ys_f1 = self.scorer.score(y*len(aug_xs), aug_xs)[-1]
                for aug_x, x_f1, y_f1 in zip(aug_xs, xs_f1, ys_f1):
                    aug_x, x_f1, y_f1 = aug_x[0], x_f1.item(), y_f1.item()
                    f1 = (x_f1 + y_f1) / 2
                    if f1 >= base_f1:
                        aug_x = self.preprocess(aug_x)
                        aug_xs_list.append(aug_x)
                        aug_ys_list += y
        xs_list = raw_xs_list + aug_xs_list
        ys_list = raw_ys_list + aug_ys_list
        # remove duplicates caused by corrector
        # print(len(xs_list), len(raw_xs_list), len(aug_xs_list))
        xs_list, ys_list = utils.remove_duplicates(xs_list, ys_list)
        # print(len(xs_list))
        return xs_list, ys_list


class General_Aug(Base_Aug):
    """docstring for General_Aug"""
    def __init__(self, config, **kwargs):
        super(General_Aug, self).__init__(config)
        self.update_config(**kwargs)
        if self.config.x_x_copy:
            self.xxcopy = XXCopy(config)
        if self.config.y_x_switch:
            self.yxswitch = YXSwitch(config)
        if self.config.ld:
            self.ld = LinearDecompose(config)
        if self.config.lc:
            self.lc = LinearCompose(config)

    def update_config(self, **kwargs):
        # update configuration accordingly
        for k,v in kwargs.items():
            setattr(self.config, k, v)

    def do_aug(self, raw_xs_list, raw_ys_list, do_copy=True):
        if do_copy:
            raw_xs_list, raw_ys_list = copy.deepcopy(raw_xs_list), copy.deepcopy(raw_ys_list)
        # augmentation
        print('Apply augmentation as follow:')
        for aug in self.config.augs:
            print(f'\t{aug}: {self.config.__dict__[aug]}')
        match self.augs:
            # none
            case (False, False, False, False):
                return raw_xs_list, raw_ys_list
            # x_x_copy
            case (True, False, False, False):
                return self.xxcopy.do_aug(raw_xs_list, raw_ys_list)
            # y_x_switch
            case (False, True, False, False):
                return self.yxswitch.do_aug(raw_xs_list, raw_ys_list)
            # linear decompose
            case (False, False, True, False):
                return self.ld.do_aug(raw_xs_list, raw_ys_list)
            # linear compose
            case (False, False, False, True):
                return self.lc.do_aug(raw_xs_list, raw_ys_list)
            # y_x_switch + linear decompose
            case (False, True, True, False):
                xs_list, ys_list = self.yxswitch.do_aug(raw_xs_list, raw_ys_list)
                xs_list, ys_list = self.ld.do_aug(xs_list, ys_list)
                xs_list, ys_list = self.yxswitch.do_aug(xs_list, ys_list)
                return xs_list, ys_list
        raise NotImplementedError
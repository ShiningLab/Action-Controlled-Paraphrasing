#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Author'
__email__ = 'Email'


# dependency
# built-in
import copy
# public
from tqdm import tqdm
from transformers import AutoTokenizer
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
            if aug_x and aug_y and aug_x != aug_y and len(aug_x) > 1 and len(aug_y) > 1:
                aug_xs_list.append(aug_x)
                aug_ys_list.append(aug_y)
        xs_list = raw_xs_list + aug_xs_list
        ys_list = raw_ys_list + aug_ys_list
        # remove duplicates
        xs_list, ys_list = utils.remove_duplicates(xs_list, ys_list)
        return xs_list, ys_list


class General_Aug(Base_Aug):
    """docstring for General_Aug"""
    def __init__(self, config, **kwargs):
        super(General_Aug, self).__init__(config)
        self.update_config(**kwargs)
        self.xxcopy = XXCopy(config)
        self.yxswitch = YXSwitch(config)
        self.ld = LinearDecompose(config)

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
            case (False, False, False):
                return raw_xs_list, raw_ys_list
            # x_x_copy
            case (True, False, False):
                return self.xxcopy.do_aug(raw_xs_list, raw_ys_list)
            # y_x_switch
            case (False, True, False):
                return self.yxswitch.do_aug(raw_xs_list, raw_ys_list)
            # linear decompose
            case (False, False, True):
                return self.ld.do_aug(raw_xs_list, raw_ys_list)
            # y_x_switch + linear decompose
            case (False, True, True):
                xs_list, ys_list = self.yxswitch.do_aug(raw_xs_list, raw_ys_list)
                xs_list, ys_list = self.ld.do_aug(xs_list, ys_list)
                xs_list, ys_list = self.yxswitch.do_aug(xs_list, ys_list)
                return xs_list, ys_list
        raise NotImplementedError
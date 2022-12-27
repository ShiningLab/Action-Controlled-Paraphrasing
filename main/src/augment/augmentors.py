#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Author'
__email__ = 'Email'


# dependency
# built-in
import copy
# public
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

class General_Aug(Base_Aug):
    """docstring for General_Aug"""
    def __init__(self, config, **kwargs):
        super(General_Aug, self).__init__(config)
        self.update_config(**kwargs)
        self.xxcopy = XXCopy(config)
        self.yxswitch = YXSwitch(config)

    def update_config(self, **kwargs):
        # update configuration accordingly
        for k,v in kwargs.items():
            setattr(self.config, k, v)

    def do_aug(self, raw_xs_list, raw_ys_list, do_copy=True):
        if do_copy:
            raw_xs_list, raw_ys_list = copy.deepcopy(raw_xs_list), copy.deepcopy(raw_ys_list)
        # augmentation
        match self.augs:
            # none
            case (False, False, False):
                return raw_xs_list, raw_ys_list
            # x_x_copy
            case (True, False, False):
                return self.xxcopy.do_aug(raw_xs_list, raw_ys_list)
            case (False, True, False):
                return self.yxswitch.do_aug(raw_xs_list, raw_ys_list)   
        raise NotImplementedError
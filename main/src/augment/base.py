#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Author'
__email__ = 'Email'


class Base_Aug(object):
    """docstring for Base_Aug"""
    def __init__(self, config):
        super(Base_Aug, self).__init__()
        self.config = config
        self.augs = tuple(self.config.__dict__[aug] for aug in self.config.augs)
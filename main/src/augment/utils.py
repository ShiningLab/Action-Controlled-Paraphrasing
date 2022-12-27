#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Author'
__email__ = 'Email'


def remove_duplicates(xs_list, ys_list):
    return map(list, zip(*list(set([(x, y) for x, y in zip(xs_list, ys_list)]))))
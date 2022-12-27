#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Author'
__email__ = 'Email'


def text_normalize(x: str) -> str:
    """
    Text normalization in preprocessing
    """
    # remove question mark
    if x.endswith('?'):
        x = x[:-1]
    # remove redundant spaces
    x = ' '.join(x.split())
    return x
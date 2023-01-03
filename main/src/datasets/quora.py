#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Author'
__email__ = 'Email'


# dependency
# built-in

# public
from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset
from tokenizers import normalizers
from tokenizers.normalizers import BertNormalizer
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
# private
from src.utils import helper


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
        self.normalizer = normalizers.Sequence([BertNormalizer()])
        if self.config.stemming:
            self.stemmer = SnowballStemmer('english')
        data_dict = helper.load_pickle(config.DATA_PKL)
        self.raw_xs = data_dict[mode]['xs']
        self.raw_ys = data_dict[mode]['ys']
        if augmenator:
            self.raw_xs, self.raw_ys = augmenator.do_aug(self.raw_xs, self.raw_ys)
        self.data_size = len(self.raw_xs)
        self.raw_xs, self.raw_ys = self.preprocess()
        if self.config.mask:
            self.xs, self.ys, self.masks = self.encode()
        else:
            self.xs, self.ys = self.encode()

    def __len__(self): 
        return self.data_size

    def __getitem__(self, idx):
        raw_x, raw_y = self.raw_xs[idx], self.raw_ys[idx]
        # bos + text + eos
        x = self.xs[idx]
        # bos + text + eos -> text + eos
        y = self.ys[idx][1:]
        # for mask control
        if self.config.mask:
            mask = self.masks[idx]
            return raw_x, raw_y, x, y, mask
        return raw_x, raw_y, x, y

    def stemming(self, x: str) -> str:
        tk_x = word_tokenize(x)
        return ' '.join([self.stemmer.stem(tk) for tk in tk_x])

    def preprocess(self):
        xs, ys = [], []
        for x, y in zip(tqdm(self.raw_xs), self.raw_ys):
            norm_x, norm_y = helper.unify_white_space(x), helper.unify_white_space(y)
            norm_x, norm_y = self.normalizer.normalize_str(norm_x), self.normalizer.normalize_str(norm_y)
            if self.config.stemming:
                norm_x, norm_y = self.stemming(norm_x), self.stemming(norm_y)
            norm_x = self.tokenizer.encode(norm_x, add_special_tokens=False, truncation=True, max_length=self.config.en_max_len)
            norm_y = self.tokenizer.encode(norm_y, add_special_tokens=False, truncation=True, max_length=self.config.de_max_len)
            norm_x, norm_y = self.tokenizer.decode(norm_x, skip_special_tokens=True), self.tokenizer.decode(norm_y, skip_special_tokens=True)
            xs.append(norm_x)
            ys.append(norm_y)
        return xs, ys

    def encode(self):
        # dict_keys(['input_ids', 'token_type_ids', 'attention_mask'])
        xs_dict = self.tokenizer.batch_encode_plus(
            self.raw_xs
            , add_special_tokens=True
            , return_tensors='pt'
            , padding=True
            , truncation=True
            # bos + text + eos
            , max_length=self.config.en_max_len + 2
         )
        ys_dict = self.tokenizer.batch_encode_plus(
            self.raw_ys
            , add_special_tokens=True
            , return_tensors='pt'
            , padding=True
            , truncation=True
            # bos + text + eos
            , max_length=self.config.de_max_len + 2
         )
        # for mask control
        if self.config.mask:
            masks = []
            for x, y in zip(xs_dict.input_ids, ys_dict.input_ids):
                shared_tokens = set(np.unique(x.numpy())) & set(np.unique(y.numpy()))
                shared_tokens.discard(self.config.bos_token_id)
                shared_tokens.discard(self.config.pad_token_id)
                mask = sum([x == t for t in shared_tokens])
                masks.append(mask)
            return xs_dict.input_ids, ys_dict.input_ids, masks
        return xs_dict.input_ids, ys_dict.input_ids

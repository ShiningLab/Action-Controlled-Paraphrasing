#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Author'
__email__ = 'Email'


# dependency
# public
import torch
from torch.utils import data as torch_data
from tqdm import tqdm
# private
from src.utils import helper
from src.utils.eval import Evaluater
from src.datasets.base import Dataset
from src.testers.base import Base_Tester


class Tester(Base_Tester):
    """docstring for Tester"""
    def __init__(self, trainer, **kwargs):
        super(Tester, self).__init__(trainer)
        self.update_config(**kwargs)
        self.setup_dataloader()

    def update_config(self, **kwargs):
        # update configuration accordingly
        for k,v in kwargs.items():
            setattr(self.config, k, v)

    def collate_fn(self, data):
        # a customized collate function used in the data loader 
        data.sort(key=len, reverse=True)
        if self.config.mask:
            raw_xs, raw_ys, xs, ys, masks = map(list, zip(*data))
            xs, ys = torch.stack(xs), torch.stack(ys)
            # apply mask strategy
            match self.config.mask_mode:
                case 'random':
                    # random mask in range [0, 1]
                    masks = [torch.randint(low=0, high=self.config.mask_size-2, size=m.shape) for m in masks]
                case '0s':
                    # all 0s mask
                    masks = [torch.full(m.shape, self.config.mask_delete_token_id) for m in masks]
                case '1s':
                    # all 1s mask
                    masks = [torch.full(m.shape, self.config.mask_keep_token_id) for m in masks]
                case '2s':
                    # all 2s mask
                    masks = [torch.full(m.shape, self.config.mask_infer_token_id) for m in masks]
                case 'oracle':
                    # cheating mask
                    pass
                case _:
                    raise NotImplementedError
            # mask padding
            masks = torch.stack(masks)
            masks = helper.pad_masks(xs, masks, self.config)
            inputs_dict = {'xs': xs, 'masks': masks}
        else:
            raise NotImplementedError
        return (raw_xs, raw_ys), inputs_dict

    def setup_dataloader(self):
        dataset = Dataset('test', self.tokenizer, self.config)
        self.dataloader = torch_data.DataLoader(
            dataset
            , batch_size=self.config.eval_batch_size
            , collate_fn=self.collate_fn
            , shuffle=False
            , num_workers=self.config.num_workers
            , pin_memory=self.config.pin_memory
            , drop_last=False
            )

    def one_epoch(self):
        # initialization
        epoch_xs, epoch_ys, epoch_ys_ = [], [], []
        # for batch in epoch
        for (raw_xs, raw_ys), inputs_dict in tqdm(self.dataloader):
            inputs_dict = {k: v.to(self.config.device) for k, v in inputs_dict.items() if v is not None}
            ys_ = self.model.generate(**inputs_dict)  # batch_size, de_max_len
            # post processing
            ys_ = ys_.cpu().detach()
            ys_ = self.tokenizer.batch_decode(ys_, skip_special_tokens=True)
            epoch_xs += raw_xs
            epoch_ys += raw_ys
            epoch_ys_ += ys_
            # break
        return float('inf'), epoch_xs, epoch_ys, epoch_ys_

    def test(self):
        self.model.eval()
        with torch.no_grad():
            for mask_mode in self.mask_modes:
                print(f'\nMask mode: {mask_mode}')
                self.config.mask_mode = mask_mode
                loss, xs, ys, ys_ = self.one_epoch()
                # evaluation
                evaluater = Evaluater(xs, ys, ys_, self.config, loss, sample=False)
                print('Epoch:{} Step:{}'.format(self.epoch, self.step) + evaluater.info)
                # random sample to show
                print(helper.show_rand_sample(xs, ys, ys_))
                # save test results
                self.test_dict[mask_mode]['eval'] = evaluater.results
                # save results
                for k, v in zip(['xs', 'ys', 'ys_'], [xs, ys, ys_]):
                    self.test_dict[mask_mode]['results'][k] = v
        helper.save_pickle(self.config.TEST_PKL, self.test_dict)
        print('Test completed.')
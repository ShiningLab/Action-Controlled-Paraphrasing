#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Author'
__email__ = 'Email'


# dependency
# built-in
import os, sys, copy, datetime, logging
# public
import torch
from torch.utils import data as torch_data
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
# private
from src.utils import helper
from src.utils.eval import Evaluater
from src.trainers.base import Base_Trainer
from src.augment.augmentors import General_Aug


# helper function
def init_logger(config):
    """initialize the logger"""
    file_handler = logging.FileHandler(filename=config.LOG_TXT)
    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    handlers = [file_handler, stdout_handler]
    logging.basicConfig(
        encoding='utf-8'
        , format='%(asctime)s | %(message)s'
        , datefmt='%Y-%m-%d %H:%M:%S'
        , level=logging.INFO
        , handlers=handlers
        )
    global logger
    logger = logging.getLogger(__name__)
    return logger


class Trainer(Base_Trainer):
    """docstring for Trainer"""
    def __init__(self, config, **kwargs):
        super(Trainer, self).__init__(config)
        self.update_config(**kwargs)
        self.initialize()

    def update_config(self, **kwargs):
        # update configuration accordingly
        for k,v in kwargs.items():
            setattr(self.config, k, v)

    def initialize(self):
        # tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.ENCODER_PATH)
        self.config.bos_token_id = self.tokenizer.cls_token_id
        self.config.eos_token_id = self.tokenizer.sep_token_id
        self.config.pad_token_id = self.tokenizer.pad_token_id
        # enable tokenizer multi-processing
        if self.config.num_workers > 0:
            os.environ["TOKENIZERS_PARALLELISM"] = "true"
        # augmenator
        self.augmenator = General_Aug(self.config)
        # dataset class
        self.Dataset = helper.get_dataset(self.config)
        # model graph
        self.model = helper.get_model(self.config).to(self.config.device)
        self.config.num_parameters = f'{sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}'
        # optimizer
        optim_params = helper.get_optim_params(self.model, self.config)
        self.optimizer = torch.optim.AdamW(optim_params, lr=self.config.learning_rate)
        # scheduler
        self.scheduler = get_linear_schedule_with_warmup(
            optimizer=self.optimizer
            , num_warmup_steps=self.config.warmup_steps
            , num_training_steps=self.config.max_steps
            )
        # restore the trainer from the checkpint if needed
        if self.config.load_ckpt:
            self.load_ckpt()
            logger.info('Trainer restored from {}.'.format(self.config.CKPT_PT))
        # save config
        self.log_dict['config'] = self.config.__dict__

    def collate_fn(self, data):
        # a customized collate function used in the data loader 
        data.sort(key=len, reverse=True)
        if self.config.mask:
            raw_xs, raw_ys, xs, ys, masks = map(list, zip(*data))
            xs, ys = torch.stack(xs), torch.stack(ys)
            # apply mask strategy
            masks = helper.process_masks(masks, self.config, self.model.training)
            # mask padding
            masks = torch.stack(masks)
            masks = helper.pad_masks(xs, masks, self.config)
            if self.mode in ['train', 'val']:
                inputs_dict = {'xs': xs, 'ys': ys, 'masks': masks}
            elif self.mode in ['test']:
                inputs_dict = {'xs': xs, 'masks': masks}
            else:
                raise NotImplementedError
        else:
            raw_xs, raw_ys, xs, ys = map(list, zip(*data))
            xs, ys = torch.stack(xs), torch.stack(ys)
            if self.mode in ['train', 'val']:
                inputs_dict = {'xs': xs, 'ys': ys}
            elif self.mode in ['test']:
                inputs_dict = {'xs': xs}
            else:
                raise NotImplementedError
        return (raw_xs, raw_ys), inputs_dict

    def setup_dataloader(self):
        train_dataset = self.Dataset('train', self.tokenizer, self.config, self.augmenator)
        train_dataloader = torch_data.DataLoader(
            train_dataset
            , batch_size=self.config.train_batch_size
            , collate_fn=self.collate_fn
            , shuffle=self.config.shuffle
            , num_workers=self.config.num_workers
            , pin_memory=self.config.pin_memory
            , drop_last=self.config.drop_last
            )
        self.dataloader_dict['train'] = train_dataloader
        # update config
        for mode in self.config.eva_modes:
            dataset = self.Dataset(mode, self.tokenizer, self.config)
            dataloader = torch_data.DataLoader(
                dataset
                , batch_size=self.config.eval_batch_size
                , collate_fn=self.collate_fn
                , shuffle=False
                , num_workers=self.config.num_workers
                , pin_memory=self.config.pin_memory
                , drop_last=False
                )
            self.dataloader_dict[mode] = dataloader
            # update config
            if mode == 'val':
                self.config.val_size = len(dataset)
            elif mode == 'test':
                self.config.test_size = len(dataset)
        # update config
        self.config.train_size = len(train_dataset)
        # to save checkpoint
        self.config.CKPT_PT = f'{self.config.train_size}_{self.config.val_size}_{self.config.test_size}.pt'
        self.config.CKPT_PT = os.path.join(self.config.CKPT_PATH, self.config.CKPT_PT)
        # to save log in txt
        self.config.LOG_TXT = f'{self.config.train_size}_{self.config.val_size}_{self.config.test_size}.txt'
        self.config.LOG_TXT = os.path.join(self.config.LOG_PATH, self.config.LOG_TXT)
        os.remove(self.config.LOG_TXT) if os.path.exists(self.config.LOG_TXT) else None
        # to save log in pickle
        self.config.LOG_PKL = f'{self.config.train_size}_{self.config.val_size}_{self.config.test_size}.pkl'
        self.config.LOG_PKL = os.path.join(self.config.LOG_PATH, self.config.LOG_PKL)
        os.remove(self.config.LOG_PKL) if os.path.exists(self.config.LOG_PKL) else None
        # initialize logger
        init_logger(self.config)
        logger.info('Initialized logger.')

    def one_epoch(self, mode):
        # initialization
        self.mode = mode
        epoch_xs, epoch_ys, epoch_ys_ = [], [], []
        # dataloader
        dataloader = tqdm(self.dataloader_dict[mode])
        if mode == 'train':
            epoch_loss, epoch_steps = 0., 0
            with logging_redirect_tqdm():
                for (raw_xs, raw_ys), inputs_dict in dataloader:
                    # move to device
                    inputs_dict = {k: v.to(self.config.device) for k, v in inputs_dict.items() if v is not None}
                    # model feedward
                    ys_, loss = self.model(**inputs_dict)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.scheduler.step()
                    self.step += 1
                    loss = loss.item()
                    dataloader.set_description('{} Loss:{:.4f} LR:{:.6f}'.format(
                        mode.capitalize(), loss, self.scheduler.get_last_lr()[0]))
                    # post processing
                    ys_ = torch.argmax(ys_, dim=-1).cpu().detach()
                    ys_ = self.tokenizer.batch_decode(ys_, skip_special_tokens=True)
                    epoch_loss += loss
                    epoch_steps += 1
                    epoch_xs += raw_xs
                    epoch_ys += raw_ys
                    epoch_ys_ += ys_
                    # break
            return epoch_loss / epoch_steps, epoch_xs, epoch_ys, epoch_ys_
        elif mode == 'val':
            epoch_loss, epoch_steps = 0., 0
            for (raw_xs, raw_ys), inputs_dict in dataloader:
                # move to device
                inputs_dict = {k: v.to(self.config.device) for k, v in inputs_dict.items() if v is not None}
                # model feedward
                ys_, loss = self.model(**inputs_dict)
                # post processing
                ys_ = torch.argmax(ys_, dim=-1).cpu().detach()
                ys_ = self.tokenizer.batch_decode(ys_, skip_special_tokens=True)
                epoch_loss += loss.item()
                epoch_steps += 1
                epoch_xs += raw_xs
                epoch_ys += raw_ys
                epoch_ys_ += ys_
                # break
            return epoch_loss / epoch_steps, epoch_xs, epoch_ys, epoch_ys_
        else:
            for (raw_xs, raw_ys), inputs_dict in dataloader:
                # move to device
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

    def train(self):
        # get dataloader
        self.setup_dataloader()
        # show configurations
        logger.info('*Configurations:*')
        for k, v in self.config.__dict__.items():
            logger.info(f'\t{k}: {v}')
        # go
        logger.info("Start training...")
        while True:
            self.epoch += 1
            self.model.train()
            loss, xs, ys, ys_ = self.one_epoch('train')
            # break
            # evaluation
            evaluater = Evaluater(xs, ys, ys_, self.config, loss)
            self.log_dict['train']['eval'] += [[self.step, self.epoch, evaluater.results]]
            logger.info('Train Epoch:{} Step:{}'.format(self.epoch, self.step) + evaluater.info)
            # random sample to show
            logger.info(helper.show_rand_sample(xs, ys, ys_))
            # validation
            if self.config.val:
                self.eval('val')
                # check if early stop
                self.early_stopping()
            # maximum training steps:
            if self.val_epoch > self.config.val_patience or self.step > self.config.max_steps:
                # do a test finally
                if self.config.test:
                    self.eval('test')
                # save log
                self.log_dict['end_time'] = datetime.datetime.now()
                helper.save_pickle(self.config.LOG_PKL, self.log_dict)
                logger.info('Log saved as {}.'.format(self.config.LOG_PKL))
                logger.info('Training completed.')
                break
            # break

    def eval(self, mode):
        self.model.eval()
        with torch.no_grad():
            loss, xs, ys, ys_ = self.one_epoch(mode)
            # evaluation
            evaluater = Evaluater(xs, ys, ys_, self.config, loss, sample=False)
            self.log_dict[mode]['eval'] += [[self.step, self.epoch, evaluater.results]]
            # save results
            for k, v in zip(['xs', 'ys', 'ys_'], [xs, ys, ys_]):
                self.results_dict[mode]['results'][k] = v
            logger.info('{} Epoch:{} Valid: {}/{} Step:{}'.format(
                mode.capitalize(), self.epoch, self.val_epoch, self.config.val_patience, self.step) + evaluater.info)
            # random sample to show
            logger.info(helper.show_rand_sample(xs, ys, ys_))

    def early_stopping(self):
        # check if early stop based on the validation
        if self.log_dict['val']['eval'][-1][-1]['keymetric'] < self.log_dict['best_val_metric']:
            logger.info(
                'Got the best validation so far! ({} < {})'.format(
                    self.log_dict['val']['eval'][-1][-1]['keymetric']
                    , self.log_dict['best_val_metric']
                    )
                )
            # update the validation best record
            self.log_dict['best_val_metric'] = self.log_dict['val']['eval'][-1][-1]['keymetric']
            # test on the best validation checkpoint
            if self.config.test:
                self.eval('test')
            # update
            for mode in self.config.eva_modes:
                # best evaluation
                self.log_dict[mode]['best_eval'] =  copy.deepcopy(self.log_dict[mode]['eval'])
                # best results
                self.results_dict[mode]['best_results'] =  copy.deepcopy(self.results_dict[mode]['results'])
            # reset validation epoch
            self.val_epoch = 0
            # save trainer
            self.save_ckpt()
            logger.info('Trainer saved as {}.'.format(self.config.CKPT_PT))
            # save log
            self.log_dict['end_time'] = datetime.datetime.now()
            helper.save_pickle(self.config.LOG_PKL, self.log_dict)
            logger.info('Log saved as {}.'.format(self.config.LOG_PKL))
        else:
            self.val_epoch += 1
#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Author'
__email__ = 'Email'


# dependency
# built-in
# import os
# public
import lightning.pytorch as pl
from transformers import AutoTokenizer
from pytorch_lightning.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, RichProgressBar
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
# private
from src import helper, model
from src.eval import Evaluator
from src.datamodule import DataModule


class S2STrainer(object):
    """docstring for S2STrainer"""
    def __init__(self, config, **kwargs):
        super(S2STrainer, self).__init__()
        self.config = config
        self.update_config(**kwargs)
        self.initialize()

    def update_config(self, **kwargs):
        # update configuration accordingly
        for k,v in kwargs.items():
            setattr(self.config, k, v)

    def initialize(self):
        # logger
        self.logger = helper.init_logger(self.config)
        self.logger.info('Logger initialized.')
        self.wandb_logger = WandbLogger(
            name=self.config.NAME
            , save_dir=self.config.LOG_PATH
            , offline=self.config.offline
            , project=self.config.PROJECT
            , log_model=self.config.log_model
            , entity=self.config.ENTITY
            , save_code=False
            , mode=self.config.wandb_mode
            )
        self.wandb_logger.experiment.config.update(self.config)
        # tokenizer
        if self.config.pretrain == 'tfg':
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.TOKENIZER_PATH)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.LM_PATH)
        self.config.bos_token_id = self.tokenizer.cls_token_id
        self.config.eos_token_id = self.tokenizer.sep_token_id
        self.config.pad_token_id = self.tokenizer.pad_token_id
        self.config.vocab_size = len(self.tokenizer)
        # model
        self.model = model.S2S(self.config, self.tokenizer)
        # datamodule
        self.dm = DataModule(self.config, self.tokenizer)
        # callbacks
        if self.config.monitor == 'val_ibleu0.8':
            filename = '{epoch}-{step}-{val_ibleu0.8:.4f}'
        else:
            raise NotImplementedError
        checkpoint_callback = ModelCheckpoint(
            dirpath=self.config.CKPT_PATH
            , filename=filename
            , monitor=self.config.monitor
            , mode='max'
            , verbose=True
            , save_last=True
            , save_top_k=1
            )
        early_stop_callback = EarlyStopping(
            monitor=self.config.monitor
            , min_delta=.0
            , patience=self.config.patience
            , verbose=True
            , mode='max'
            )
        # early stopping trainer
        self.trainer = pl.Trainer(
            accelerator=self.config.accelerator
            , precision=self.config.precision
            , logger = self.wandb_logger
            , callbacks=[checkpoint_callback, early_stop_callback, RichProgressBar()]
            , fast_dev_run=self.config.fast_dev_run
            , max_epochs=self.config.max_epochs
            , val_check_interval=self.config.val_check_interval
            , check_val_every_n_epoch=self.config.check_val_every_n_epoch
            , num_sanity_val_steps=self.config.num_sanity_val_steps
            , gradient_clip_val=self.config.gradient_clip_val
            , deterministic=True
            , inference_mode=True
            , profiler=self.config.profiler if self.config.profiler else None
            )

    def train(self):
        self.logger.info('*Configurations:*')
        for k, v in self.config.__dict__.items():
            self.logger.info(f'\t{k}: {v}')
        self.logger.info('Start training...')
        
        # ckpt_path = 'last'
        # if self.config.load_ckpt:
            # ckpt_path = './res/ckpts/qqp/w2a_w2r/w2a_w2r/w2a_w2r/gpt2/0/epoch=36-step=115625-val_bleu4=0.4561.ckpt'
        
        self.trainer.fit(
            model=self.model
            , datamodule=self.dm
            , ckpt_path='last' if self.config.load_ckpt else None
            )
        # testing
        self.logger.info('Start testing...')

        # predict_dict = self.predict(
            # ckpt_path=os.path.join(self.config.CKPT_PATH, 'epoch=36-step=115625-val_bleu4=0.8494.ckpt')
            # )

        predict_dict = self.predict(ckpt_path='best')
        # evaluation
        eva = Evaluator(predict_dict, self.config)
        self.logger.info(eva.info)
        # save results
        helper.save_pickle(predict_dict, self.config.RESULTS_PKL)

        # predict_dict = helper.load_pickle(self.config.RESULTS_PKL)

        self.logger.info('Results saved as {}.'.format(self.config.RESULTS_PKL))
        # upload to wandb
        self.update_wandb(predict_dict, 'test')
        self.logger.info('Done.')

    def predict(self, dataloaders=None, ckpt_path=None):
        outputs = self.trainer.predict(
            model=self.model
            , dataloaders=dataloaders if dataloaders else None
            , datamodule=None if dataloaders else self.dm
            , ckpt_path=ckpt_path
            , return_predictions=True
            )
        # reformat batches into a dict
        outputs_dict = dict()
        for k in outputs[0]:
            outputs_dict[k] = helper.flatten_list([d[k] for d in outputs])
        return outputs_dict

    def update_wandb(self, update_dict, key='test'):
        for k in update_dict:
            dtype = type(update_dict[k][0])
            if dtype == str:
                pass
            elif dtype == int:
                update_dict[k] = [str(v) for v in update_dict[k]]
            elif dtype == list:
                update_dict[k] = [','.join(v) for v in update_dict[k]]
            else:
                raise NotImplementedError
            dsize = len(update_dict[k])
        self.wandb_logger.log_text(
            key=key
            , columns=list(update_dict.keys())
            , data=[[update_dict[k][i] for k in update_dict] for i in range(dsize)]
            )
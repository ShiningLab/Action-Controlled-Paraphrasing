#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Author'
__email__ = 'Email'


# built-in
import copy
# public
import torch.nn as nn
from transformers import AutoConfig, EncoderDecoderConfig, EncoderDecoderModel
# private
from .lm import LAN_MODELS

class ModelGraph(nn.Module):
    """docstring for ModelGraph"""
    def __init__(self, config, **kwargs):
        super(ModelGraph, self).__init__()
        # initialize configurations
        self.config = config
        # update configurations
        self.update_config(**kwargs)
        # encoder
        encoder_config = AutoConfig.from_pretrained(self.config.ENCODER_PATH)
        encoder_config.hidden_size = self.config.hidden_size
        encoder_config.num_hidden_layers = self.config.num_hidden_layers
        encoder_config.num_attention_heads = self.config.num_attention_heads
        encoder_config.intermediate_size = self.config.intermediate_size
        # decoder
        decoder_config = AutoConfig.from_pretrained(self.config.DECODER_PATH)
        decoder_config.hidden_size = self.config.hidden_size
        decoder_config.num_hidden_layers = self.config.num_hidden_layers
        decoder_config.num_attention_heads = self.config.num_attention_heads
        decoder_config.intermediate_size = self.config.intermediate_size
        decoder_config.is_decoder = True
        decoder_config.add_cross_attention = True
        decoder_config.decoder_start_token_id = self.config.bos_token_id
        # seq2seq
        tfm_config = EncoderDecoderConfig.from_encoder_decoder_configs(
            encoder_config=encoder_config
            , decoder_config=decoder_config
            )
        tfm_config.vocab_size = decoder_config.vocab_size
        tfm_config.pad_token_id = self.config.pad_token_id
        tfm_config.decoder_start_token_id = self.config.bos_token_id
        # initialize model
        self.tfm = EncoderDecoderModel(config=tfm_config)
        self.config.model_config = self.tfm.config
        self.config.en_vocab_size = self.config.model_config.encoder.vocab_size
        self.config.de_vocab_size = self.config.model_config.decoder.vocab_size
        # mask embedding layer
        if self.config.mask:
            self.mask_embeddings = nn.Embedding(
                num_embeddings=self.config.mask_size
                , embedding_dim=self.config.hidden_size
                )

    def update_config(self, **kwargs):
        # update configuration accordingly
        for k,v in kwargs.items():
            setattr(self.config, k, v)

    def forward(self, xs, ys, masks=None):
        """
        xs (Tensor): batch_size, en_seq_len
        ys (Tensor): batch_size, de_seq_len
        masks (Tensor): batch_size, en_seq_len
        """
        if masks is not None:
            # batch_size, en_seq_len, hidden_size
            inputs_embeds = self.tfm.encoder.embeddings.word_embeddings(xs)
            masks_embeds = self.mask_embeddings(masks)
            inputs_embeds += masks_embeds
            ys_ = self.tfm(labels=ys, inputs_embeds=inputs_embeds)
        else:
            ys_ = self.tfm(input_ids=xs, labels=ys)
        # logits: batch_size, de_seq_len, de_vocab_size
        # loss: 1
        return ys_.logits, ys_.loss

    def generate(self, xs, masks=None):
        if masks is not None:
            # batch_size, en_seq_len, hidden_size
            inputs_embeds = self.tfm.encoder.embeddings.word_embeddings(xs)
            masks_embeds = self.mask_embeddings(masks)
            inputs_embeds += masks_embeds
            return self.tfm.generate(
                inputs_embeds =inputs_embeds
                , max_new_tokens=self.config.de_max_len
                , num_beams=self.config.num_beams
                , early_stopping=True
                , pad_token_id=self.config.pad_token_id
                , bos_token_id=self.config.bos_token_id
                , eos_token_id=self.config.eos_token_id
                )
        else:
            return self.tfm.generate(
                input_ids =xs
                , max_new_tokens=self.config.de_max_len
                , num_beams=self.config.num_beams
                , early_stopping=True
                , pad_token_id=self.config.pad_token_id
                , bos_token_id=self.config.bos_token_id
                , eos_token_id=self.config.eos_token_id
                )
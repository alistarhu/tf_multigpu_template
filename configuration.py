#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os

'''
Default configurations of model train and test
'''
#####
LOG_DIR = 'result_exp'  # where checkpoints, logs are saved
RUN_NAME = 'hl_test1'  # identifier of the experiment
train_sample_num = 256133
gpu_ids = '0,1,2,3'
gpu_num = 4
multi_thread = 6#multi-thread num -->$ lscpu
prefetch_capacity = 5
epoch_num = 20 #50
batch_size = 64
#####

MODEL_CONFIG = {
  'embed_config': {
                   'init_method': 'kaiming_normal',
                   'use_bn': True,
                   'bn_scale': True,
                   'bn_momentum': 0.05,
                   'bn_epsilon': 1e-6,
                   'embedding_feature_num': 256,
                   'weight_decay': 5e-4,
                   'net_choose': 'alex',#select backbone type
                    },

}

TRAIN_CONFIG = {
    'train_dir': os.path.join(LOG_DIR, RUN_NAME),
    'config_saver_dir': os.path.join(LOG_DIR, RUN_NAME, 'config_json'),
    'checkpoint_dir': os.path.join(LOG_DIR, RUN_NAME, 'checkpoints'),
    'log_dir': os.path.join(LOG_DIR, RUN_NAME, 'log'),#save for tensorboard
    'gpu_select': gpu_ids,# select which gups to run train, single:'0' or multi:'0,2,4'

    'seed': 123,  # fix seed for reproducing experiments

    # config of input train and validate data
    'train_data_config': {
                        'img_label_list_path': 'data_txt/train_list.txt',
                        'preprocessing_name': 'data_argu1',#add data-argument for train data
                        'num_examples_per_epoch': train_sample_num,#total train samples count
                        'epoch': epoch_num,#train epochs nums
                        'batch_size': batch_size,
                        'prefetch_threads': multi_thread,#use multi-thread for load in data
                        'prefetch_capacity': prefetch_capacity*gpu_num,# prefetch m batches
                        },

    'validation_data_config': {
                        'img_label_list_path': 'data_txt/validate_list.txt',
                        'preprocessing_name': 'None',#without data-argument for train data
                        'batch_size': batch_size,
                        'prefetch_threads': 1,#use multi-thread for load in data
                        'prefetch_capacity': prefetch_capacity*gpu_num,# prefetch m batches
                        },

    # Optimizer for training the model.
    'optimizer_config': {'optimizer': 'MOMENTUM',  # SGD / MOMENTUM / Adam are supported
                       'momentum': 0.9,
                       'use_nesterov': False, },

    # Learning rate configs
    'lr_config': {'policy': 'exponential',# piecewise_constant / exponential / cosine
                'initial_lr': 0.01,
                'num_epochs_per_decay': 1,
                'lr_decay_factor': 0.8685113737513527,
                'staircase': True, },

    # Frequency at which loss and global step are logged
    'log_every_n_steps': 10,

    # Update tensorboard-summary every n steps
    'tensorboard_summary_every_n_steps': 100,

    # Frequency to save model
    'save_model_every_n_step': train_sample_num // (batch_size*gpu_num),  # save model every epoch

    # How many model checkpoints to keep. No limit if None.
    'max_checkpoints_to_keep': 30,# save last 30 epochs
}




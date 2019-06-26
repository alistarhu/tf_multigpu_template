#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
import tensorflow as tf
slim = tf.contrib.slim
CURRENT_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(CURRENT_DIR, '..'))


import netdefs.net_def as mobilenet


#build model
class ModelConstruct:
    def __init__(self, model_config, train_config, img_in, label, mode='train'):
        self.model_config = model_config
        self.train_config = train_config
        self.mode = mode
        assert mode in ['train', 'validation', 'inference']

        self.dataloader = None
        self.logits = None
        self.image_instances = img_in
        self.label = label

        # self.batch_loss = None
        self.total_loss = None
        self.init_fn = None
        self.global_step = None


    def is_training(self):
        return self.mode == 'train'


    # Can edit here to use your own network!!!
    def build_net_forward(self, reuse=False):
        config = self.model_config['embed_config']
        #if config['net_choose'] == 'alex':

        logits, pred = mobilenet.mobilenetv2_addBias(self.image_instances,
                                                     num_classes=5,
                                                     channel_rito=1,
                                                     is_train=self.is_training(),
                                                     reuse=reuse)
        self.logits = logits


    #define the loss function
    def build_loss(self):

        loss_ = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.label, logits=self.logits))
        l2_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))  # L2 regularization
        total_loss = loss_ + l2_loss

        correct_pred = tf.equal(tf.argmax(self.logits, 1), tf.cast(self.label, tf.int64))
        acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        self.total_loss = total_loss
        self.acc = acc
        # with tf.device("/cpu:0"):
        #     tf.summary.scalar('total_loss', self.total_loss, family=self.mode)
        #     tf.summary.scalar('acc', self.acc, family=self.mode)



    def setup_global_step(self):
        global_step = tf.Variable(initial_value=0, name='global_step',trainable=False,
                                  collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])
        self.global_step = global_step


    #outer interface, to do a set operation for build embed net
    def build(self, reuse=False):
        #Creates all ops for training and evaluation
        with tf.name_scope(self.mode):
            #self.build_inputs()
            self.build_net_forward(reuse=reuse)

            if self.mode in ['train', 'validation']:
                self.build_loss()
            # if self.is_training():
            #     self.setup_global_step()#set global steps
#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import logging
import tensorflow as tf
import numpy as np
import random

CURRENT_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(CURRENT_DIR, '..'))
from datasets.sampler import Sampler
from datasets.transforms import Compose, RandomGray, RandomCrop, CenterCrop, RandomStretch



class DataLoader(object):
    def __init__(self, config, is_training=False):
        self.config = config
        self.img_list = None
        self.label_list = None
        #preprocess_name = get(config, 'preprocessing_name', None)
        preprocess_name = config['preprocessing_name']
        logging.info('preproces -- {}'.format(preprocess_name))

        img_list = []
        label_list = []
        img_label_txt = config['img_label_list_path']
        if os.path.exists(img_label_txt):
            with open(img_label_txt, 'r') as f:
                for line in f:
                    img_path_, label_ = line.strip().split(',')
                    img_list.append(img_path_)
                    label_list.append(int(label_))
        else:
            raise ValueError('Load {} fail, please generate image label list first!!!'.format(img_label_txt))
        self.img_list = img_list
        self.label_list = label_list
        self.img_type = img_list[0].split('.')[-1].upper()

        # add data-argument to train data
        if preprocess_name == 'data_argu1':
            # Compose() a set of function, Composes several transforms together,
            # self.img_transform = Compose([RandomStretch(),
            #                             CenterCrop((size_x+8, size_x+8)),
            #                             RandomCrop(size_x), CenterCrop((size_z, size_z))])
            self.img_transform = None

        # without data-argument to train data
        elif preprocess_name == 'None':
            self.img_transform = None
        else:
            raise ValueError('Preprocessing name {} was not recognized.'.format(preprocess_name))

        self.sampler = Sampler(img_list, shuffle=is_training)# Generate a idx list of data_source


    def build_dataset(self):
        def sample_generator():
            for video_id in self.sampler:
                sample = self.img_list[video_id]
                label = self.label_list[video_id]
                yield sample, label

        def transform_fn(param1, param2):
            img_path = param1
            label = param2
            img_file_raw = tf.read_file(img_path)
            # image_ = tf.image.decode_jpeg(img_file_raw, channels=3, dct_method="INTEGER_ACCURATE")
            #image_ = tf.image.decode_image(img_file_raw, channels=3)
            image_ = tf.image.decode_bmp(img_file_raw, channels=3)
            if self.img_transform is not None:
                # instance_image = self.x_transform(instance_image)
                image_, _, _ = self.img_transform(image_)
            return image_, label

        # load in videos(frame list)
        dataset = tf.data.Dataset.from_generator(sample_generator,
                                                 output_types=(tf.string, tf.int32),
                                                 output_shapes=(tf.TensorShape([]), tf.TensorShape([])))
        dataset = dataset.repeat()
        dataset = dataset.map(transform_fn, num_parallel_calls=self.config['prefetch_threads'])
        dataset = dataset.batch(self.config['batch_size'])
        dataset = dataset.prefetch(self.config['prefetch_capacity'])
        # dataset.prefetch(m*batch_size)  dataset.batch(batch_size) <--> dataset.batch(batch_size) dataset.prefetch(m)

        self.dataset_tf = dataset


    def build_iterator(self):
        self.iterator = self.dataset_tf.make_one_shot_iterator()

    def build(self):
        self.build_dataset()
        self.build_iterator()

    def get_one_batch(self):
        return self.iterator.get_next()


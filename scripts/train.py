#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
import tensorflow as tf
import numpy as np
import time
from datetime import datetime
import logging
import glob

CURRENT_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(CURRENT_DIR, '..'))
import configuration
from netdefs.model_construct import ModelConstruct
from datasets.dataloader import DataLoader
#import tfrecord_write_read as tfwr
#import net_def
from scripts.utils import save_cfgs

model_config = configuration.MODEL_CONFIG
train_config = configuration.TRAIN_CONFIG


logging.getLogger().setLevel(logging.INFO)

def load(sess, saver, checkpoint_dir):
    import re
    print("[*] Reading checkpoints...")

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
        counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
        print("[*] Success to read {}".format(ckpt_name))
        return True, counter
    else:
        print("[*] Failed to find a checkpoint")
        return False, 0

# set the learning_rate according to    configuration.py->TRAIN_CONFIG->lr_config
def _configure_learning_rate(train_config, global_step):
    train_data_config = train_config['train_data_config']
    num_batches_per_epoch = int(train_data_config['num_examples_per_epoch']/train_data_config['batch_size'])

    lr_config = train_config['lr_config']
    if lr_config['policy'] == 'piecewise_constant':
        lr_boundaries = [int(e * num_batches_per_epoch) for e in lr_config['lr_boundaries']]
        return tf.train.piecewise_constant(global_step, lr_boundaries, lr_config['lr_values'])
    elif lr_config['policy'] == 'exponential':
        decay_steps = int(num_batches_per_epoch) * lr_config['num_epochs_per_decay']
        return tf.train.exponential_decay(lr_config['initial_lr'], global_step, decay_steps=decay_steps,
                                          decay_rate=lr_config['lr_decay_factor'], staircase=lr_config['staircase'])
    elif lr_config['policy'] == 'cosine':
        T_total = train_config['train_data_config']['epoch'] * num_batches_per_epoch
        return 0.5 * lr_config['initial_lr'] * (1 + tf.cos(np.pi * tf.to_float(global_step) / T_total))
    else:
        raise ValueError('Learning rate policy [%s] was not recognized', lr_config['policy'])


# set the optimizer according to  configuration.py->TRAIN_CONFIG->optimizer_config
def _configure_optimizer(train_config, learning_rate):
    optimizer_config = train_config['optimizer_config']
    optimizer_name = optimizer_config['optimizer'].upper()
    if optimizer_name == 'MOMENTUM':
        optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=optimizer_config['momentum'],
                                                use_nesterov=optimizer_config['use_nesterov'], name='Momentum')
    elif optimizer_name == 'SGD':
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    elif optimizer_name == 'ADAM':
        optimizer = tf.train.AdamOptimizer(learning_rate)
    else:
        raise ValueError('Optimizer [%s] was not recognized', optimizer_config['optimizer'])
    return optimizer


def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in list(zip(*tower_grads)):
        grads = []
        for g, _ in grad_and_vars:
            expand_g = tf.expand_dims(g, 0)
            grads.append(expand_g)
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def main():
    # Select gpu to run
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = train_config['gpu_select']
    gpu_list = [int(i) for i in train_config['gpu_select'].split(',')]

    # Create training directory
    os.makedirs(train_config['train_dir'], exist_ok=True)
    os.makedirs(train_config['checkpoint_dir'], exist_ok=True)
    os.makedirs(train_config['log_dir'], exist_ok=True)
    os.makedirs(train_config['config_saver_dir'], exist_ok=True)

    # Save configurations .json in train_dir
    save_cfgs(train_config['config_saver_dir'], model_config, train_config)

    g = tf.Graph()
    with g.as_default():
        global_step = tf.Variable(initial_value=0, name='global_step', trainable=False,
                                  collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])
        learning_rate = _configure_learning_rate(train_config, global_step)  # set lr
        tf.summary.scalar('learning_rate', learning_rate)#see learning rate in tensorboard-scalars
        opt = _configure_optimizer(train_config, learning_rate)  # set optimizer
        tower_grads = []#gradient list of each gpu

        # Build dataloader
        ## train dataloader
        train_data_config = train_config['train_data_config']
        with tf.device("/cpu:0"):
            train_dataloader = DataLoader(train_data_config, is_training=True)
            train_dataloader.build()
        ## validate dataloader
        validate_data_config = train_config['validation_data_config']
        with tf.device("/cpu:0"):
            validate_dataloader = DataLoader(validate_data_config, is_training=False)
            validate_dataloader.build()

        #Build network on multi-gpu with the same graph
        with tf.variable_scope(tf.get_variable_scope()):
            for i in range(len(gpu_list)):
                #build same graph on each gpu
                with tf.device('/gpu:%d'%i):
                    # to distinguish variable in different gpu
                    with tf.name_scope('model_%d'%i):
                        if i==0:
                            train_image_instances, train_label = train_dataloader.get_one_batch()
                            train_image_instances = tf.to_float(train_image_instances)
                            train_label = tf.to_int32(train_label)
                            model = ModelConstruct(model_config, train_config, train_image_instances, train_label,
                                                   mode='train')
                            model.build()
                            with tf.device("/cpu:0"):
                                tf.summary.scalar('total_loss', model.total_loss, family='train')
                                tf.summary.scalar('acc', model.acc, family='train')
                            #validate
                            validate_image_instances, validate_label = validate_dataloader.get_one_batch()
                            validate_image_instances = tf.to_float(validate_image_instances)
                            validate_label = tf.to_int32(validate_label)
                            model_va = ModelConstruct(model_config, train_config, validate_image_instances, validate_label,
                                                      mode='validation')
                            model_va.build(reuse=True)
                            with tf.device("/cpu:0"):
                                tf.summary.scalar('total_loss', model_va.total_loss, family='validation')
                                tf.summary.scalar('acc', model_va.acc, family='validation')
                        else:
                            train_image_instances, train_label = train_dataloader.get_one_batch()
                            train_image_instances = tf.to_float(train_image_instances)
                            train_label = tf.to_int32(train_label)
                            model = ModelConstruct(model_config, train_config, train_image_instances, train_label,
                                                   mode='train')
                            model.build(reuse=True)

                        tf.get_variable_scope().reuse_variables()
                        grad = opt.compute_gradients(model.total_loss)
                        tower_grads.append(grad)


        mean_grads = average_gradients(tower_grads)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = opt.apply_gradients(mean_grads, global_step=global_step)

        # save checkpoint
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=train_config['max_checkpoints_to_keep'])
        # save the graph
        summary_writer = tf.summary.FileWriter(train_config['log_dir'], g)
        summary_op = tf.summary.merge_all()

        global_variables_init_op = tf.global_variables_initializer()
        local_variables_init_op = tf.local_variables_initializer()
        g.finalize()  # Finalize graph to avoid adding ops by mistake

        # Dynamically allocate GPU memory
        gpu_options = tf.GPUOptions(allow_growth=True)
        sess_config = tf.ConfigProto(gpu_options=gpu_options)

        sess = tf.Session(config=sess_config)
        model_path = tf.train.latest_checkpoint(train_config['checkpoint_dir'])
        # re-train or start new train
        if not model_path:
            sess.run(global_variables_init_op)
            sess.run(local_variables_init_op)
            start_step = 0
        else:
            logging.info('Restore from last checkpoint: {}'.format(model_path))
            sess.run(local_variables_init_op)
            saver.restore(sess, model_path)
            start_step = tf.train.global_step(sess, global_step.name) + 1

        # Training loop
        start_time = time.time()
        data_config = train_config['train_data_config']
        total_steps = int(data_config['epoch'] * data_config['num_examples_per_epoch'] / (data_config['batch_size']*configuration.gpu_num))
        logging.info('Train for {} steps'.format(total_steps))
        for step in range(start_step, total_steps):
            _, loss = sess.run([train_op, model.total_loss,  ])
            if step % train_config['log_every_n_steps'] == 0:
                logging.info('{}-->step {:d} - ({:.2f}%), total loss = {:.2f} '.format(datetime.now(), step, float(step) / total_steps * 100, loss))
            # each 100 steps update tensorboard-summay
            if step % train_config['tensorboard_summary_every_n_steps'] == 0:
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, step)
            # save model each epoch
            if step % train_config['save_model_every_n_step'] == 0 or (step + 1) == total_steps:
                saver.save(sess, os.path.join(train_config['checkpoint_dir'], 'model.ckpt'),global_step=step)
        duration = time.time() - start_time
        m, s = divmod(duration, 60)
        h, m = divmod(m, 60)
        print('The total training loop finished after {:d}h:{:02d}m:{:02d}s'.format(int(h), int(m), int(s)))
        sess.close()





if __name__=='__main__':
    main()

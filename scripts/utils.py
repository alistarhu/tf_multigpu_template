#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json

def save_cfgs(save_dir, model_config, train_config):
    #Save all configurations in JSON format
    with open(os.path.join(save_dir, 'model_config.json'), 'w') as f:
        json.dump(model_config, f, indent=2)
    with open(os.path.join(save_dir, 'train_config.json'), 'w') as f:
        json.dump(train_config, f, indent=2)


def load_cfgs(checkpoint):
  if os.path.isdir(checkpoint):
    train_dir = checkpoint
  else:
    train_dir = os.path.dirname(checkpoint)
  with open(os.path.join(train_dir, 'model_config.json'), 'r') as f:
    model_config = json.load(f)
  with open(os.path.join(train_dir, 'train_config.json'), 'r') as f:
    train_config = json.load(f)
  return model_config, train_config
# Author: Arun Prabhu (arun.rajendra.prabhu@iais.fraunhofer.de) 10/11/2019

"""SqueezeDet Model configuration for inference"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import joblib
from utils import util
from easydict import EasyDict as edict
import numpy as np
import tensorflow as tf
from nn_skeleton import ModelSkeleton

class SqueezeDet_inf(ModelSkeleton):
  def __init__(self, mc, gpu_id=0):
    with tf.device('/gpu:{}'.format(gpu_id)):
      ModelSkeleton.__init__(self, mc)

      self._add_forward_graph()

  def _add_forward_graph(self):
    """NN architecture."""

    mc = self.mc
    freeze_weight = mc.TRAIN_ONLY_LAST_LAYER
    num_mask_params = 4
    if mc.EIGHT_POINT_REGRESSION:
      num_mask_params = 8
    if mc.LOAD_PRETRAINED_MODEL:
      assert tf.gfile.Exists(mc.PRETRAINED_MODEL_PATH), \
          'Cannot find pretrained model at the given path:' \
          '  {}'.format(mc.PRETRAINED_MODEL_PATH)
      self.caffemodel_weight = joblib.load(mc.PRETRAINED_MODEL_PATH)

    self.image_input = tf.placeholder(
        tf.float32, [None, None, None, 3],
        name='image_input'
    )
    conv1 = self._conv_layer(
        'conv1', self.image_input, filters=64, size=3, stride=2,
        padding='SAME', freeze=True)
    pool1 = self._pooling_layer(
        'pool1', conv1, size=3, stride=2, padding='VALID')

    fire2 = self._fire_layer(
        'fire2', pool1, s1x1=16, e1x1=64, e3x3=64, freeze=freeze_weight)
    fire3 = self._fire_layer(
        'fire3', fire2, s1x1=16, e1x1=64, e3x3=64, freeze=freeze_weight)
    pool3 = self._pooling_layer(
        'pool3', fire3, size=3, stride=2, padding='VALID')

    fire4 = self._fire_layer(
        'fire4', pool3, s1x1=32, e1x1=128, e3x3=128, freeze=freeze_weight)
    fire5 = self._fire_layer(
        'fire5', fire4, s1x1=32, e1x1=128, e3x3=128, freeze=freeze_weight)
    pool5 = self._pooling_layer(
        'pool5', fire5, size=3, stride=2, padding='VALID')

    fire6 = self._fire_layer(
        'fire6', pool5, s1x1=48, e1x1=192, e3x3=192, freeze=freeze_weight)
    fire7 = self._fire_layer(
        'fire7', fire6, s1x1=48, e1x1=192, e3x3=192, freeze=freeze_weight)
    fire8 = self._fire_layer(
        'fire8', fire7, s1x1=64, e1x1=256, e3x3=256, freeze=freeze_weight)
    fire9 = self._fire_layer(
        'fire9', fire8, s1x1=64, e1x1=256, e3x3=256, freeze=freeze_weight)

    # Two extra fire modules that are not trained before
    fire10 = self._fire_layer(
        'fire10', fire9, s1x1=96, e1x1=384, e3x3=384, freeze=freeze_weight)
    fire11 = self._fire_layer(
        'fire11', fire10, s1x1=96, e1x1=384, e3x3=384, freeze=freeze_weight)
    # dropout11 = tf.nn.dropout(fire11, self.keep_prob, name='drop11')

    num_output = mc.ANCHOR_PER_GRID * (mc.CLASSES + 1 + num_mask_params)
    self.preds = self._conv_layer(
        'conv12', fire11, filters=num_output, size=3, stride=1,
        padding='SAME', xavier=False, relu=False, stddev=0.0001)

  def _fire_layer(self, layer_name, inputs, s1x1, e1x1, e3x3, stddev=0.01,
      freeze=False):
    """Fire layer constructor.

    Args:
      layer_name: layer name
      inputs: input tensor
      s1x1: number of 1x1 filters in squeeze layer.
      e1x1: number of 1x1 filters in expand layer.
      e3x3: number of 3x3 filters in expand layer.
      freeze: if true, do not train parameters in this layer.
    Returns:
      fire layer operation.
    """

    sq1x1 = self._conv_layer(
        layer_name+'/squeeze1x1', inputs, filters=s1x1, size=1, stride=1,
        padding='SAME', stddev=stddev, freeze=freeze)
    ex1x1 = self._conv_layer(
        layer_name+'/expand1x1', sq1x1, filters=e1x1, size=1, stride=1,
        padding='SAME', stddev=stddev, freeze=freeze)
    ex3x3 = self._conv_layer(
        layer_name+'/expand3x3', sq1x1, filters=e3x3, size=3, stride=1,
        padding='SAME', stddev=stddev, freeze=freeze)

    return tf.concat([ex1x1, ex3x3], 3, name=layer_name+'/concat')

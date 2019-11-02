# Author: Arun Prabhu (arun.rajendra.prabhu@iais.fraunhofer.de) 10/11/2019

"""VGG Model configuration for cityscape dataset"""

import numpy as np

from .config import base_model_config

def cityscape_vgg16_config(mask_parameterization):
  """Specify the parameters to tune below."""
  mc                       = base_model_config('CITYSCAPE')

  mc.IMAGE_WIDTH           = 1024
  mc.IMAGE_HEIGHT          = 512
  mc.BATCH_SIZE            = 5

  mc.WEIGHT_DECAY          = 0.0001
  mc.LEARNING_RATE         = 0.01
  mc.DECAY_STEPS           = 10000
  mc.MAX_GRAD_NORM         = 1.0
  mc.MOMENTUM              = 0.9
  mc.LR_DECAY_FACTOR       = 0.5

  mc.LOSS_COEF_BBOX        = 5.0
  mc.LOSS_COEF_CONF_POS    = 75.0
  mc.LOSS_COEF_CONF_NEG    = 100.0
  mc.LOSS_COEF_CLASS       = 1.0

  mc.PLOT_PROB_THRESH      = 0.4
  mc.NMS_THRESH            = 0.4
  mc.PROB_THRESH           = 0.005
  mc.TOP_N_DETECTION       = 64

  mc.DATA_AUGMENTATION     = True
  mc.DRIFT_X               = 150
  mc.DRIFT_Y               = 100
  mc.EXCLUDE_HARD_EXAMPLES = False

  mc.ANCHOR_BOX            = set_anchors(mc)
  mc.ANCHORS               = len(mc.ANCHOR_BOX)
  mc.ANCHOR_PER_GRID       = 9
  if mask_parameterization == 8:
    mc.EIGHT_POINT_REGRESSION = True

  return mc

def set_anchors(mc):
  H, W, B = 31, 63, 9
  anchor_shapes = np.reshape(
      [np.array(
          [[15.93, 16.07], [34.82, 37.27], [34.32, 70.14],
           [102.69, 66.61], [55.86, 122.26], [172.25, 116.73],
           [96.53, 202.02], [273.93, 190.09], [196.36, 315.79]])] * H * W,
      (H, W, B, 2)
  )
  center_x = np.reshape(
      np.transpose(
          np.reshape(
              np.array([np.arange(1, W+1)*16]*H*B), 
              (B, H, W)
          ),
          (1, 2, 0)
      ),
      (H, W, B, 1)
  )
  center_y = np.reshape(
      np.transpose(
          np.reshape(
              np.array([np.arange(1, H+1)*16]*W*B),
              (B, W, H)
          ),
          (2, 1, 0)
      ),
      (H, W, B, 1)
  )
  anchors = np.reshape(
      np.concatenate((center_x, center_y, anchor_shapes), axis=3),
      (-1, 4)
  )

  return anchors
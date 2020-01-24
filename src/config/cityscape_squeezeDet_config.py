# Author: Arun Prabhu (arun.rajendra.prabhu@iais.fraunhofer.de) 10/11/2019

"""SqueezeDet Model configuration for cityscape dataset"""

import numpy as np

from .config import base_model_config

def cityscape_squeezeDet_config(mask_parameterization, log_anchors, tune_only_last_layer):
  """Specify the parameters to tune below."""
  mc                       = base_model_config('CITYSCAPE')

  mc.IMAGE_WIDTH           = 1024
  mc.IMAGE_HEIGHT          = 512
  mc.BATCH_SIZE            = 10

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

  mc.ANCHOR_BOX            = set_anchors(mc, log_anchors)
  mc.ANCHORS               = len(mc.ANCHOR_BOX)
  mc.ANCHOR_PER_GRID       = 9
  if mask_parameterization == 8:
    mc.EIGHT_POINT_REGRESSION = True

  mc.TRAIN_ONLY_LAST_LAYER = False
  if tune_only_last_layer:
    mc.TRAIN_ONLY_LAST_LAYER = True
  return mc

def set_anchors(mc, log_anchors):
  H, W, B = 31, 63, 9
  if log_anchors:
    print("Using Log domain extracted anchors")
    anchor_shapes = np.reshape(
      [np.array(
          [[8.01, 11.25], [11.45, 26.49], [18.02, 13.88],
           [21.40, 50.10], [31.07, 24.21], [42.67, 103.73],
           [55.73, 42.22], [107.92, 76.43], [171.29, 181.58]])] * H * W,
      (H, W, B, 2)
    )
  else:
    print("Using Spatial domain extracted anchors")
    anchor_shapes = np.reshape(
        [np.array(
            [[17.31, 18.20], [35.13, 39.49], [99.93, 66.42],
             [34.60, 73.31], [56.66, 125.19], [166.94, 114.14],
             [94.15, 203.37], [257.57, 187.70], [196.69, 312.63]])] * H * W,
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

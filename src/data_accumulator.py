# Author: Arun Rajendra Prabhu (arun.rajendra.prabhu@iais.fraunhofer.de) 10/11/2019

"""Accumulator script.

This file implements code using which, the ground-truth and detection files are generated for 
the PASCAL VOC evaluator from a folder containing checkpoints.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import time
import sys
import os
import json
import glob
import math

import numpy as np
import tensorflow as tf

from config import *
from train import _draw_box
from nets import *
from dataset.cityscape_utils.cityscapesscripts.helpers.labels import assureSingleInstanceName

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'checkpoint', './data/model_checkpoints/squeezeDet/model.ckpt-87000',
    """Path to the model parameter file.""")
tf.app.flags.DEFINE_string(
    'input_path', './data/sample.png',
    """Input image or video to be detected. Can process glob input such as """
    """./data/00000*.png.""")
tf.app.flags.DEFINE_string(
    'out_dir', './data/out/', """Directory to dump output image or video.""")
tf.app.flags.DEFINE_string(
    'demo_net', 'squeezeDet', """Neural net architecture.""")
tf.app.flags.DEFINE_string(
    'label_path', 'None', """Label path.""")

def get_bounding_box_parameterization(polygon, height, width):
  """Extract the bounding box of a polygon representing the instance mask.
  Args:
      polygon: a list of points representing the instance mask.
      height: height of the image
      width: width of the image
  Returns:
    mask_vector: bounding box of the instance mask [xmin, ymin, xmax, ymax].
  """
  outline = np.array(polygon)
  rr, cc = outline[:,1], outline[:,0]
  xmin = max(min(cc), 0)
  xmax = min(max(cc), width-1)
  ymin = max(min(rr), 0)
  ymax = min(max(rr), height-1)
  width       = xmax - xmin
  height      = ymax - ymin
  center_x  = xmin + 0.5*width 
  center_y  = ymin + 0.5*height
  mask_vector = [xmin, ymin, xmax, ymax, center_x, center_y, width, height]
  return mask_vector

def _load_cityscape_annotations( _label_path, index, include_8_point_masks=False, threshold=10):
  """Load the cityscape instance segmentation annotations.
  Args: include_8_point_masks: a boolean representing if we need to extract 8 point mask parameterization
        threshold: a threshold to filter objects whose width or height are less than threshold
  Returns:
    idx2annotation: dictionary mapping image name to the bounding box parameters
    idx2polygons: dictionary mapping image name to the raw binary mask polygons or None depending on include_8_point_masks. 
  """
  bboxes = []
  boundaryadhesions = []
  polygons = []
  left_margin = 6
  right_margin = 5
  top_margin = 5
  bottom_margin = 5
  filename = os.path.join(_label_path, index[:-15]+'gtFine_polygons.json')
  print("Label path", filename)
  permitted_classes = sorted(['person', 'rider', 'car', 'truck', 'bus', 'motorcycle', 'bicycle'])
  _class_to_idx = dict(zip(permitted_classes, range(len(permitted_classes))))
  with open(filename) as f:
    data_dict = json.load(f)
    imgHeight = data_dict['imgHeight']
    imgWidth = data_dict['imgWidth']
    instances = data_dict['objects']
    for instance in instances:
      class_name = instance['label']
      modified_name = assureSingleInstanceName(class_name, reject_groups=True)
      if modified_name != None and modified_name in permitted_classes:
        polygon = np.array(instance['polygon'], dtype=np.float)
        cls = _class_to_idx[modified_name]
        vector = get_bounding_box_parameterization(polygon, imgHeight, imgWidth)
        xmin, ymin, xmax, ymax, cx, cy, w, h = vector
        if w >= threshold or h >= threshold:
          # Filter objects which are less than threshold
          assert xmin >= 0.0 and xmin <= xmax, \
              'Invalid bounding box x-coord xmin {} or xmax {} at {}.txt' \
                  .format(xmin, xmax, index)
          assert ymin >= 0.0 and ymin <= ymax, \
              'Invalid bounding box y-coord ymin {} or ymax {} at {}.txt' \
                  .format(ymin, ymax, index)
          bboxes.append([cx, cy, w, h, cls])
          # Since we use only box to determine boundaryadhesion, it is common for ,
          # both 8 and 4 point
          if include_8_point_masks:
            boundaryadhesion = [0]*8
          else:
            boundaryadhesion = [0]*4
          # Not mutually exclusive
          if cx - (w/2) <= left_margin:
            boundaryadhesion[0] = True
          if cy - (h/2) <= top_margin:
            boundaryadhesion[1] = True
          if cx + (w/2) >= (imgWidth-1-right_margin):
            boundaryadhesion[2] = True
          if cy + (h/2) >= (imgHeight-1-bottom_margin):
            boundaryadhesion[3] = True

          if include_8_point_masks:
            # Derived adhesions
            if cx - (w/2) <= left_margin or cy - (h/2) <= top_margin:
              boundaryadhesion[4] = True
            if cy + (h/2) >= (imgHeight-1-bottom_margin) or cx - (w/2) <= left_margin:
              boundaryadhesion[5] = True
            if cx + (w/2) >= (imgWidth-1-right_margin) or cy + (h/2) >= (imgHeight-1-bottom_margin):
              boundaryadhesion[6] = True
            if cy - (h/2) <= top_margin or cx + (w/2) >= (imgWidth-1-right_margin):
              boundaryadhesion[7] = True
          boundaryadhesions.append(boundaryadhesion)
          if include_8_point_masks:
            polygons.append([imgHeight, imgWidth, polygon])
  return bboxes, boundaryadhesions, polygons

def decode_parameterization(mask_vector):
  """Decodes the octagonal parameterization of the mask to get
     the 8 points approximation of the polygon.
  Args:
    mask vector: [cx, cy, w, h, of1, of2, of3, of4]
  Returns:
    intersecting_pts: list of points where the octagonal mask 
                      intersects the bounding box
  """

  def _get_intersecting_point(vert_hor, eq1, pt, m):
    """Finds the point of intersection of a line1 with vertical or 
       horizontal line.
    Args:
      vert_hor: "vert" or "hor" represents intersection is to be found 
                with respect to which line.
      eq1: equation of vertical or horizontal line.
      pt : point on line1.
      m  : slope of line1.
    Returns:
      point of intersection of line1 with vertical or horizontal.
    """
    pt_x, pt_y = pt
    c = pt_y - (m*pt_x)
    if vert_hor == "vert":
        x_cor = eq1
        y_cor = (m*x_cor) + c
    else:
        y_cor = eq1
        x_cor = (y_cor - c)/m
    return (x_cor, y_cor)

  center_x, center_y, width, height, off1, off2, off3, off4 = mask_vector
  cos = math.cos(math.radians(45))
  sin = math.cos(math.radians(45))
  pts = [0,0,0,0]
  pts[0] = (center_x-off1*cos, center_y-off1*sin)
  pts[1] = (center_x-off2*cos, center_y+off2*sin)
  pts[2] = (center_x+off3*cos, center_y+off3*sin)
  pts[3] = (center_x+off4*cos, center_y-off4*sin)
  xmin = center_x - (0.5*width)
  xmax = center_x + (0.5*width)
  ymin = center_y - (0.5*height)
  ymax = center_y + (0.5*height)
  points = [pts[0], pts[1], pts[1], pts[2], pts[2], pts[3], pts[3], pts[0]]
  eq1s = [xmin, xmin, ymax, ymax, xmax, xmax, ymin, ymin]
  vert_or_hors = ["vert", "vert", "hor", "hor", "vert", "vert", "hor", "hor"]
  EPSILON = 1e-8
  dr1 = (pts[2][0]-pts[0][0])
  if dr1 == 0:
    dr1 = EPSILON
  dr2 = (pts[3][0]-pts[1][0])
  if dr2 == 0:
    dr2 = EPSILON
  m1 = (pts[2][1]-pts[0][1])/dr1
  m2 = (pts[3][1]-pts[1][1])/dr2
  assert m1 != 0 and m2 != 0, "Slopes are zero in _get_intersecting_point"
  ms = [-1/m1, -1/m2, -1/m2, -1/m1, -1/m1, -1/m2, -1/m2, -1/m1]
  intersecting_pts = []
  for i, eq1, pt, vert_hor, m in zip(range(8), eq1s, points, vert_or_hors, ms):
      op_pt = _get_intersecting_point(vert_hor, eq1, pt, m)
      x_val, y_val = op_pt
      if vert_hor == 'vert':
        if i == 0 or i == 5:
          y_val = min(y_val, center_y)
        elif i == 1 or i == 4:
          y_val = max(y_val, center_y)
      else:
        if i == 2 or i == 7:
          x_val = min(x_val, center_x)
        elif i == 3 or i == 6:
          x_val = max(x_val, center_x)
      op_pt = (min(max(xmin, x_val), xmax), min(max(ymin, y_val), ymax))
      intersecting_pts.append(op_pt)
  return intersecting_pts

def image_demo(label_path, mask_parameterization_now, log_anchors_now, encoding_type_now, checkpoint_path, out_dir, fmt):
  assert FLAGS.demo_net == 'squeezeDet' or FLAGS.demo_net == 'squeezeDet+', \
      'Selected neural net architecture not supported: {}'.format(FLAGS.demo_net)

  with tf.Graph().as_default():
    if FLAGS.demo_net == 'squeezeDet':
      mc = cityscape_squeezeDet_config(mask_parameterization_now, log_anchors_now, False, encoding_type_now)
      mc.BATCH_SIZE = 1
      # model parameters will be restored from checkpoint
      mc.LOAD_PRETRAINED_MODEL = False
      model = SqueezeDet(mc)
    elif FLAGS.demo_net == 'squeezeDet+':
      mc = cityscape_squeezeDetPlus_config(mask_parameterization_now, log_anchors_now, False, encoding_type_now)
      mc.BATCH_SIZE = 1
      mc.LOAD_PRETRAINED_MODEL = False
      model = SqueezeDetPlus(mc)

    saver = tf.train.Saver(model.model_params)

    image_preprocess_time = []
    forward_pass = []
    post_processing = []
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
      saver.restore(sess, checkpoint_path)

      for f in glob.iglob(FLAGS.input_path):
        start_1 = time.time()
        im = cv2.imread(f)
        orig_h, orig_w, _ = [float(v) for v in im.shape]
        im = im.astype(np.float32, copy=False)
        im = cv2.resize(im, (mc.IMAGE_WIDTH, mc.IMAGE_HEIGHT))
        input_image = im - mc.BGR_MEANS
        image_preprocess_time.append(time.time()-start_1)

        file_name = os.path.split(f)[1]
        gt_bboxes, _, gt_polys = _load_cityscape_annotations(label_path, file_name, mask_parameterization_now == 8)
        # gt_polys if [(y1,x1), (y2,x2), ...]
        if np.shape(gt_bboxes)[0] == 0:
          print("Skipping:", file_name)
          continue
        gt_bbox = np.asarray(gt_bboxes)[:,0:4]
        gt_labels = np.asarray(gt_bboxes)[:,4]
        # Scale the boxes.
        x_scale = mc.IMAGE_WIDTH/orig_w
        y_scale = mc.IMAGE_HEIGHT/orig_h
        gt_bbox[:, 0::2] = gt_bbox[:, 0::2]*x_scale
        gt_bbox[:, 1::2] = gt_bbox[:, 1::2]*y_scale

        # Detect
        start_2 = time.time()
        det_boxes, det_probs, det_class = sess.run(
            [model.det_boxes, model.det_probs, model.det_class],
            feed_dict={model.image_input:[input_image], model.keep_prob: 1.0})

        forward_pass.append(time.time()-start_2)
        # Filter
        start_3 = time.time()
        final_boxes, final_probs, final_class = model.filter_prediction(
            det_boxes[0], det_probs[0], det_class[0])

        keep_idx    = [idx for idx in range(len(final_probs)) \
                          if final_probs[idx] > mc.PLOT_PROB_THRESH]
        final_boxes = [final_boxes[idx] for idx in keep_idx]
        final_probs = [final_probs[idx] for idx in keep_idx]
        final_class = [final_class[idx] for idx in keep_idx]
        post_processing.append(time.time()-start_3)

        out_file_name = os.path.join(out_dir, 'out_'+file_name)
        with open(os.path.join(out_dir, "groundtruths", file_name+".txt"), 'w') as f:
          for u in range(len(gt_labels)):
            if fmt == 'xywh':
              f.write(
                '{:s} {:.2f} {:.2f} {:.2f} {:.2f}\n'.format(
                    mc.CLASS_NAMES[int(gt_labels[u])].lower(), gt_bbox[u][0]-(gt_bbox[u][2]/2), gt_bbox[u][1]-(gt_bbox[u][3]/2), gt_bbox[u][2], gt_bbox[u][3]) #L T w h
              )
            elif fmt == 'xyrb':
              f.write(
                '{:s} {:.2f} {:.2f} {:.2f} {:.2f}\n'.format(
                    mc.CLASS_NAMES[int(gt_labels[u])].lower(), gt_bbox[u][0]-(gt_bbox[u][2]/2), gt_bbox[u][1]-(gt_bbox[u][3]/2), gt_bbox[u][0]+(gt_bbox[u][2]/2), gt_bbox[u][1]+(gt_bbox[u][3]/2)) #L T R B
              )
            else:
              gt_p = gt_polys[u][2]
              xmin1 = float(max(min(gt_p[:,0]), 0))*x_scale
              ymin1 = float(max(min(gt_p[:,1]), 0))*y_scale
              xmax1 = float(min(max(gt_p[:,0]), 2048-1))*x_scale
              ymax1 = float(min(max(gt_p[:,1]), 1024-1))*y_scale

              gt_p[:,0] = gt_p[:,0]*x_scale
              gt_p[:,1] = gt_p[:,1]*y_scale

              xmin2, ymin2, xmax2, ymax2 = gt_bbox[u][0]-(gt_bbox[u][2]/2), gt_bbox[u][1]-(gt_bbox[u][3]/2), gt_bbox[u][0]+(gt_bbox[u][2]/2), gt_bbox[u][1]+(gt_bbox[u][3]/2)
              assert abs(xmin1 - xmin2) <= 1.0, "GT Error in xmin "+str(xmin1)+" "+str(xmin2) 
              assert abs(ymin1 - ymin2) <= 1.0, "GT Error in ymin "+str(ymin1)+" "+str(ymin2)
              assert abs(xmax1 - xmax2) <= 1.0, "GT Error in xmax "+str(xmax1)+" "+str(xmax2)
              assert abs(ymax1 - ymax2) <= 1.0, "GT Error in ymax "+str(ymax1)+" "+str(ymax2)

              write_str = mc.CLASS_NAMES[int(gt_labels[u])].lower() 
              for ele in gt_p:
                write_str += " "
                write_str += str(round(ele[0],2))
                write_str += " "
                write_str += str(round(ele[1],2))
              write_str += "\n"
              f.write(write_str)
        f.close()
        with open(os.path.join(out_dir, "detections", file_name+".txt"), 'w') as f:
          for u in range(len(final_class)):
            if fmt == 'xywh':
              f.write(
                '{:s} {:.6f} {:.2f} {:.2f} {:.2f} {:.2f}\n'.format(
                    mc.CLASS_NAMES[final_class[u]].lower(), final_probs[u], final_boxes[u][0]-(final_boxes[u][2]/2), final_boxes[u][1]-(final_boxes[u][3]/2), final_boxes[u][2], final_boxes[u][3]) #L T w h
              )
            elif fmt == 'xyrb':
              f.write(
                '{:s} {:.6f} {:.2f} {:.2f} {:.2f} {:.2f}\n'.format(
                    mc.CLASS_NAMES[final_class[u]].lower(), final_probs[u], final_boxes[u][0]-(final_boxes[u][2]/2), final_boxes[u][1]-(final_boxes[u][3]/2), final_boxes[u][0]+(final_boxes[u][2]/2), final_boxes[u][1]+(final_boxes[u][3]/2)) #L T R B
              )
            else:
              xmin2, ymin2, xmax2, ymax2 = final_boxes[u][0]-(final_boxes[u][2]/2), final_boxes[u][1]-(final_boxes[u][3]/2), final_boxes[u][0]+(final_boxes[u][2]/2), final_boxes[u][1]+(final_boxes[u][3]/2)
              dt_p = decode_parameterization(final_boxes[u])
              dt_p = np.asarray(dt_p)
              xmin1 = max(min(dt_p[:,0]), 0)
              ymin1 = max(min(dt_p[:,1]), 0)
              xmax1 = min(max(dt_p[:,0]), 1024-1)
              ymax1 = min(max(dt_p[:,1]), 512-1)
              assert abs(xmin1 - xmin2) <= 1.0, "DT Error in xmin "+str(xmin1)+" "+str(xmin2) 
              assert abs(ymin1 - ymin2) <= 1.0, "DT Error in ymin "+str(ymin1)+" "+str(ymin2)
              assert abs(xmax1 - xmax2) <= 1.0, "DT Error in xmax "+str(xmax1)+" "+str(xmax2)
              assert abs(ymax1 - ymax2) <= 1.0, "DT Error in ymax "+str(ymax1)+" "+str(ymax2)

              write_str = mc.CLASS_NAMES[final_class[u]].lower()
              write_str += " "
              write_str += str(round(final_probs[u], 6))
              for ele in dt_p:
                write_str += " "
                write_str += str(round(ele[0],2))
                write_str += " "
                write_str += str(round(ele[1],2))
              write_str += "\n"
              f.write(write_str)

        f.close()
        print ('Image detection output saved to {}'.format(out_file_name))
        # break

      with open(os.path.join(out_dir, 'runtime.txt'), 'w') as f:
        for prep_time, fp_time, pp_time in zip(image_preprocess_time, forward_pass, post_processing):
          f.write(
              '{:s} {:.6f} {:s} {:.6f} {:s} {:.6f}\n'.format(
              'preprocessing: ', prep_time, ' forward pass: ', fp_time, ' post processing: ', pp_time)
          )
      f.close


def main(argv=None):
  if not tf.gfile.Exists(FLAGS.out_dir):
    tf.gfile.MakeDirs(FLAGS.out_dir)
  checkpoints = []
  if 'model.ckpt' in FLAGS.checkpoint:
    checkpoints.append(FLAGS.checkpoint)
  else:
    checkpoints.extend([os.path.join(x[0], 'model.ckpt-200000') for x in os.walk(FLAGS.checkpoint) if 'inference' not in x[0] and 'model.ckpt-200000.index' in x[2]])
  # for frmt in ['xywh', 'xyrb', 'coords']:
  for frmt in ['xywh', 'coords']:
    for i, c in enumerate(checkpoints):
      print("\nProcessing:", c.split('\\')[-2], "Output_format", frmt)
      if '4' in c and '-4' not in c:
          print("mask_parameters_now:", 4)
          mask_param = 4
      else:
          print("mask_parameters_now:", 8)
          mask_param = 8
      if '_lin_lin_anch' in c:
          print("encoding_type_now: asymmetric_linear")
          print('log_anchors_now:', False)
          encoding_scheme = 'asymmetric_linear'
          use_log_anchors = False
      elif '_log_log_anch' in c:
          print("encoding_type_now: asymmetric_log")
          print("log_anchors_now:", True)
          encoding_scheme = 'asymmetric_log'
          use_log_anchors = True
      else:
          print("encoding_type_now: normal")
          encoding_scheme = 'normal'
          if 'spatial' in c:
              print("log_anchors_now:", False)
              use_log_anchors = False
          else:
              print("log_anchors_now:", True)
              use_log_anchors = True

      if mask_param == 4 and frmt == 'coords':
        continue
      log_dir = os.path.join(FLAGS.out_dir, str(c.split('\\')[-3]), str(c.split('\\')[-2])+"_"+frmt)
      if not tf.gfile.Exists(log_dir):
        tf.gfile.MakeDirs(log_dir)

      if not tf.gfile.Exists(os.path.join(log_dir, "detections")):
        tf.gfile.MakeDirs(os.path.join(log_dir, "detections"))
      if not tf.gfile.Exists(os.path.join(log_dir, "groundtruths")):
        tf.gfile.MakeDirs(os.path.join(log_dir, "groundtruths"))

      image_demo(FLAGS.label_path, mask_param, use_log_anchors, encoding_scheme, c, log_dir, frmt)

if __name__ == '__main__':
    tf.app.run()

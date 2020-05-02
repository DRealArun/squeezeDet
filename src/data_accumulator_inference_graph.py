# Author: Bichen Wu (bichen@berkeley.edu) 08/25/2016

"""SqueezeDet Demo.

In image detection mode, for a given image, detect objects and draw bounding
boxes around them. In video detection mode, perform real-time detection on the
video stream.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import time
import sys
import os
import glob
import json
import math
import numpy as np
import tensorflow as tf
from scipy import special as sp
from config import *
import copy
from train import _viz_prediction_result, _draw_box
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

def run_inference_on_multiple_images(image_path_list, graph, label_path, mask_parameterization_now):
  output_dict_list = []
  read_images = []
  time_diff = []
  gt_bounding_boxes = []
  gt_classes = []
  file_names = []
  gt_polygons = []
  with graph.as_default():
    with tf.Session() as sess:
      # Get handles to input and output tensors
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in ['conv12/bias_add']:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          print(tensor_name)
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_input_1:0')
      # Run inference
      for image_path in image_path_list:
        print("Processing", image_path)
        f_name = os.path.split(image_path)[1]
        gt_bboxes, _, gt_polys = _load_cityscape_annotations(label_path, f_name, mask_parameterization_now == 8)
        if np.shape(gt_bboxes)[0] == 0:
          print("Skipping:", f_name)
          continue
        gt_bbox = np.asarray(gt_bboxes)[:,0:4]
        gt_labels = np.asarray(gt_bboxes)[:,4]

        file_names.append(image_path)
        gt_bounding_boxes.append(gt_bbox)
        gt_polygons.append(gt_polys)
        gt_classes.append(gt_labels)
        image_np = cv2.imread(image_path)
        BGR_MEANS = np.array([[[103.939, 116.779, 123.68]]])
        image_np = image_np.astype(np.float32, copy=False)
        image_np = cv2.resize(image_np, (1024, 512))
        image_unexpanded = image_np - BGR_MEANS
        read_images.append(image_np)
        image = np.expand_dims(image_unexpanded, axis=0)
        time_start = time.time()
        output_dict = sess.run(tensor_dict,
                               feed_dict={image_tensor: image})
        output_dict_list.append(output_dict)
        time_diff.append(time.time()-time_start)
        # break
  return output_dict_list, read_images, time_diff, gt_bounding_boxes, gt_classes, file_names, gt_polygons

def set_anchors(H, W, log_anchors=False):
  B = 9
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
      print("Using Linear domain extracted anchors")
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

def safe_exp(w, thresh):
  slope = np.exp(thresh)
  lin_bool = w > thresh
  lin_region = np.where(lin_bool, np.ones_like(w), np.zeros_like(w))
  lin_out = slope*(w - thresh + 1.)
  exp_out = np.exp(np.where(lin_bool, np.zeros_like(w), w))
  out = lin_region*lin_out + (1.-lin_region)*exp_out
  return out

def sigmoid(input_volume):
  return (1 / (1 + np.exp(-input_volume)))

def iou(box1, box2):
  """Compute the Intersection-Over-Union of two given boxes.

  Args:
    box1: array of 4 elements [cx, cy, width, height].
    box2: same as above
  Returns:
    iou: a float number in range [0, 1]. iou of the two boxes.
  """
  lr = min(box1[0]+0.5*box1[2], box2[0]+0.5*box2[2]) - \
      max(box1[0]-0.5*box1[2], box2[0]-0.5*box2[2])
  if lr > 0:
    tb = min(box1[1]+0.5*box1[3], box2[1]+0.5*box2[3]) - \
        max(box1[1]-0.5*box1[3], box2[1]-0.5*box2[3])
    if tb > 0:
      intersection = tb*lr
      union = box1[2]*box1[3]+box2[2]*box2[3]-intersection
      return intersection/union
  return 0

def batch_iou(boxes, box):
  """Compute the Intersection-Over-Union of a batch of boxes with another
  box.

  Args:
    box1: 2D array of [cx, cy, width, height].
    box2: a single array of [cx, cy, width, height]
  Returns:
    ious: array of a float number in range [0, 1].
  """
  lr = np.maximum(
      np.minimum(boxes[:,0]+0.5*boxes[:,2], box[0]+0.5*box[2]) - \
      np.maximum(boxes[:,0]-0.5*boxes[:,2], box[0]-0.5*box[2]),
      0
  )
  tb = np.maximum(
      np.minimum(boxes[:,1]+0.5*boxes[:,3], box[1]+0.5*box[3]) - \
      np.maximum(boxes[:,1]-0.5*boxes[:,3], box[1]-0.5*box[3]),
      0
  )
  inter = lr*tb
  union = boxes[:,2]*boxes[:,3] + box[2]*box[3] - inter
  return inter/union

def nms(boxes, probs, threshold):
  """Non-Maximum supression.
  Args:
    boxes: array of [cx, cy, w, h] (center format)
    probs: array of probabilities
    threshold: two boxes are considered overlapping if their IOU is largher than
        this threshold
    form: 'center' or 'diagonal'
  Returns:
    keep: array of True or False.
  """
  order = probs.argsort()[::-1] 
  keep = [True]*len(order)
  # count = 0
  for i in range(len(order)-1):
    ovps = batch_iou(boxes[order[i+1:]], boxes[order[i]])
    for j, ov in enumerate(ovps):
      # count +=1
      if ov > threshold:
        keep[order[j+i+1]] = False
  return keep

def soft_nms(boxes, probs):
  """Soft Non-Maximum supression.
  Args:
    boxes: array of [cx, cy, w, h] (center format)
    probs: array of probabilities
    threshold: two boxes are considered overlapping if their IOU is largher than
        this threshold
    form: 'center' or 'diagonal'
  Returns:
    recalculated_probs: probabilities recalculated
  """
  order = probs.argsort()[::-1]
  new_probs = copy.deepcopy(probs)
  # count = 0
  for i in range(len(order)-1):
    ovps = batch_iou(boxes[order[i+1:]], boxes[order[i]])
    new_probs[order[i+1:]] *= np.exp(-1*(ovps**2)/0.5)
  # print(count)
  return new_probs

def filter_prediction(boxes, probs, cls_idx, PROB_THRESH, softnms=True):
  filtered_idx = np.nonzero(probs> PROB_THRESH)[0]
  probs = probs[filtered_idx]
  boxes = boxes[filtered_idx]
  cls_idx = cls_idx[filtered_idx]

  final_boxes = []
  final_probs = []
  final_cls_idx = []
  NUM_CLASSES = 7
  if softnms:
      final_boxes = boxes
      final_probs = soft_nms(boxes, probs)
      final_cls_idx = cls_idx
  else:
      for c in range(NUM_CLASSES):
        idx_per_class = [i for i in range(len(probs)) if cls_idx[i] == c]
        keep = nms(boxes[idx_per_class], probs[idx_per_class], 0.4)
        for i in range(len(keep)):
          if keep[i]:
            final_boxes.append(boxes[idx_per_class[i]])
            final_probs.append(probs[idx_per_class[i]])
            final_cls_idx.append(c)
  return final_boxes, final_probs, final_cls_idx

    
def interpret_output(output_volume, mask_parameterization=4, img_id='default', log_anchors=False, imgSize=(1024,512), encoding_type_now='normal'):
  ANCHOR_PER_GRID = 9
  NUM_CLASSES = 7
  BATCH_SIZE, H, W, nc = np.shape(output_volume)
  ANCHOR_BOX = set_anchors(H, W, log_anchors)
  ANCHORS = len(ANCHOR_BOX)
  num_class_probs = ANCHOR_PER_GRID*NUM_CLASSES
  pred_class_probs = np.reshape(
        sp.softmax(
            np.reshape(
                output_volume[:, :, :, :num_class_probs],
                [-1, NUM_CLASSES]
            ), axis=1 
        ), [BATCH_SIZE, ANCHORS, NUM_CLASSES])
  
  num_confidence_scores = ANCHOR_PER_GRID+num_class_probs
  pred_conf = sp.expit(
    np.reshape(
        output_volume[:, :, :, num_class_probs:num_confidence_scores],
        [BATCH_SIZE, ANCHORS]
    )
  )
  pred_box_delta = np.reshape(
    output_volume[:, :, :, num_confidence_scores:],
    [BATCH_SIZE, ANCHORS, mask_parameterization]
  )
  
  anchor_x = ANCHOR_BOX[:, 0]
  anchor_y = ANCHOR_BOX[:, 1]
  anchor_w = ANCHOR_BOX[:, 2]
  anchor_h = ANCHOR_BOX[:, 3]

  delta_x, delta_y, delta_w, delta_h = np.squeeze(pred_box_delta[:,:,0]), np.squeeze(pred_box_delta[:,:,1]), np.squeeze(pred_box_delta[:,:,2]), np.squeeze(pred_box_delta[:,:,3])


  if mask_parameterization == 8:
    delta_of1, delta_of2, delta_of3, delta_of4 = np.squeeze(pred_box_delta[:,:,4]), np.squeeze(pred_box_delta[:,:,5]), np.squeeze(pred_box_delta[:,:,6]), np.squeeze(pred_box_delta[:,:,7])
    EPSILON = 1e-8
    anchor_diag = (anchor_w**2 + anchor_h**2)**(0.5)
    box_of1 = (anchor_diag * safe_exp(delta_of1, 1.0))-EPSILON
    box_of2 = (anchor_diag * safe_exp(delta_of2, 1.0))-EPSILON
    box_of3 = (anchor_diag * safe_exp(delta_of3, 1.0))-EPSILON
    box_of4 = (anchor_diag * safe_exp(delta_of4, 1.0))-EPSILON

  if encoding_type_now == 'normal':
    # Normal decoding
    box_center_x = anchor_x + delta_x * anchor_w
    box_center_y = anchor_y + delta_y * anchor_h
    box_width = anchor_w * safe_exp(delta_w, 1.0)
    box_height = anchor_h * safe_exp(delta_h, 1.0)
    # box_width = anchor_w * np.exp(delta_w)
    # box_height = anchor_h * np.exp(delta_h)
    xmins= box_center_x-box_width/2
    ymins = box_center_y-box_height/2
    xmaxs = box_center_x+box_width/2
    ymaxs = box_center_y+box_height/2

  elif encoding_type_now == 'asymmetric_log':
    # Asymmetric Log
    EPSILON = 0.5
    delta_xmin, delta_ymin, delta_xmax, delta_ymax = delta_x, delta_y, delta_w, delta_h
    xmins = anchor_x - (anchor_w * (np.exp(delta_xmin)-EPSILON))
    ymins = anchor_y - (anchor_h * (np.exp(delta_ymin)-EPSILON))
    xmaxs = anchor_x + (anchor_w * (np.exp(delta_xmax)-EPSILON))
    ymaxs = anchor_y + (anchor_h * (np.exp(delta_ymax)-EPSILON))
  elif encoding_type_now == 'asymmetric_linear':
    # Asymmetric Lìnear
    delta_xmin, delta_ymin, delta_xmax, delta_ymax = delta_x, delta_y, delta_w, delta_h
    xmins_a = anchor_x-anchor_w/2
    ymins_a = anchor_y-anchor_h/2
    xmaxs_a = anchor_x+anchor_w/2
    ymaxs_a = anchor_y+anchor_h/2
    xmins = xmins_a + delta_xmin * anchor_w
    ymins = ymins_a + delta_ymin * anchor_h
    xmaxs = xmaxs_a + delta_xmax * anchor_w
    ymaxs = ymaxs_a + delta_ymax * anchor_h

  # Trimming
  xmins = np.minimum(np.maximum([0.0]*len(xmins), xmins), [imgSize[0]-1.0]*len(xmins))
  ymins = np.minimum(np.maximum([0.0]*len(ymins), ymins), [imgSize[1]-1.0]*len(ymins))
  xmaxs = np.maximum(np.minimum([imgSize[0]-1.0]*len(xmaxs), xmaxs), [0.0]*len(xmaxs))
  ymaxs = np.maximum(np.minimum([imgSize[1]-1.0]*len(ymaxs), ymaxs), [0.0]*len(ymaxs))
  width       = xmaxs - xmins
  height      = ymaxs - ymins
  box_center_x  = xmins + 0.5*width 
  box_center_y  = ymins + 0.5*height
  box_width  = width
  box_height  = height

  probs = np.multiply(
      pred_class_probs,
      np.reshape(pred_conf, [BATCH_SIZE, ANCHORS, 1])
  )
  det_probs = np.amax(probs, 2)
  det_class = np.argmax(probs, 2)
  if mask_parameterization == 8:
    det_boxes = np.transpose(np.stack([box_center_x, box_center_y, box_width, box_height, box_of1, box_of2, box_of3, box_of4]))
  else:
    det_boxes = np.transpose(np.stack([box_center_x, box_center_y, box_width, box_height]))
  return det_boxes, np.squeeze(det_probs), np.squeeze(det_class)

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

def image_demo_inference_graph(label_path, mask_parameterization_now, log_anchors_now, encoding_type_now, \
  checkpoint_path, out_dir, fmt, softnms=False):
  assert FLAGS.demo_net == 'squeezeDet', 'Selected neural net architecture not supported: {}'.format(FLAGS.demo_net)
  CLASS_NAMES = tuple(sorted(('person', 'rider', 'car', 'truck', 'bus', 'motorcycle', 'bicycle')))
  _class_to_idx = dict(zip(CLASS_NAMES, range(len(CLASS_NAMES))))
  TEST_IMAGE_PATHS = glob.iglob(os.path.join(FLAGS.input_path))
  detection_graph = tf.Graph()
  with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(checkpoint_path, 'rb') as fid:
      serialized_graph = fid.read()
      od_graph_def.ParseFromString(serialized_graph)
      tf.import_graph_def(od_graph_def, name='')

  outputs, img_read, time_values, gt_bounding_boxes, gt_classes, filenames, gt_polys = run_inference_on_multiple_images(TEST_IMAGE_PATHS, \
                                                                                        detection_graph, label_path, mask_parameterization_now)
  int_times = []
  total_times =[]
  nms_times = []
  post_times = []
  IMAGE_WIDTH = 1024
  IMAGE_HEIGHT = 512
  x_scale = IMAGE_WIDTH/2048
  y_scale = IMAGE_HEIGHT/1024
  PLOT_PROB_THRESH = 0.4
  PROB_THRESH = 0.005
  counter = 0
  for i, output_dict, image_np, run_time, gt_labels, gt_bbox in zip(filenames, outputs, img_read, time_values, gt_classes, gt_bounding_boxes):
    total_time = 0
    total_time += run_time*1000
    interpret_start = time.time()
    boxes, probs, classes = interpret_output(output_dict['conv12/bias_add'], mask_parameterization_now, i.split('\\')[-1][:-4], log_anchors_now, [IMAGE_WIDTH, IMAGE_HEIGHT], encoding_type_now)
    interpret_stop = time.time()
    int_time = (interpret_stop-interpret_start)*1000
    int_times.append(int_time)
    total_time += int_time
    NMS_start = time.time()
    det_bbox, det_prob, det_class = filter_prediction(boxes, probs, classes, PROB_THRESH, softnms)
    NMS_stop = time.time()
    nms_time = (NMS_stop-NMS_start)*1000
    nms_times.append(nms_time)
    total_time += nms_time
    postprocess_start = time.time()
    keep_idx    = [idx for idx in range(len(det_prob)) \
                      if det_prob[idx] >= PLOT_PROB_THRESH]
    final_boxes = [det_bbox[idx] for idx in keep_idx]
    final_probs = [det_prob[idx] for idx in keep_idx]
    final_class = [det_class[idx] for idx in keep_idx]
    postprocess_stop = time.time()
    postprocess_time = (postprocess_stop-postprocess_start)*1000
    post_times.append(postprocess_time)
    # _draw_box(
    #     image_np, final_boxes,
    #     [CLASS_NAMES[idx]+': (%.2f)'% prob \
    #         for idx, prob in zip(final_class, final_probs)],
    #     (0, 0, 255), draw_masks=False, fill=False)
    # out_file_name = os.path.join(out_dir, i.split('\\')[-1][:-4]+".png")
    # cv2.imwrite(out_file_name, image_np)

    total_time += postprocess_time
    total_times.append(total_time)
    gt_bbox = np.asarray(gt_bbox)
    gt_bbox[:, 0::2] = gt_bbox[:, 0::2]*x_scale
    gt_bbox[:, 1::2] = gt_bbox[:, 1::2]*y_scale
    with open(os.path.join(out_dir, "groundtruths", i.split('\\')[-1][:-4]+".txt"), 'w') as f:
      for u in range(len(gt_labels)):
        if fmt == 'xywh':
          f.write(
            '{:s} {:.2f} {:.2f} {:.2f} {:.2f}\n'.format(
                CLASS_NAMES[int(gt_labels[u])].lower(), gt_bbox[u][0]-(gt_bbox[u][2]/2), gt_bbox[u][1]-(gt_bbox[u][3]/2), gt_bbox[u][2], gt_bbox[u][3]) #L T w h
          )
        elif fmt == 'xyrb':
          f.write(
            '{:s} {:.2f} {:.2f} {:.2f} {:.2f}\n'.format(
                CLASS_NAMES[int(gt_labels[u])].lower(), gt_bbox[u][0]-(gt_bbox[u][2]/2), gt_bbox[u][1]-(gt_bbox[u][3]/2), gt_bbox[u][0]+(gt_bbox[u][2]/2), gt_bbox[u][1]+(gt_bbox[u][3]/2)) #L T R B
          )
        else:
          gt_p = gt_polys[counter][u][2]
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

          write_str = CLASS_NAMES[int(gt_labels[u])].lower() 
          for ele in gt_p:
            write_str += " "
            write_str += str(round(ele[0],2))
            write_str += " "
            write_str += str(round(ele[1],2))
          write_str += "\n"
          f.write(write_str)
    f.close()
    with open(os.path.join(out_dir, "detections", i.split('\\')[-1][:-4]+".txt"), 'w') as f:
      for u in range(len(final_class)):
        if fmt == 'xywh':
          f.write(
            '{:s} {:.6f} {:.2f} {:.2f} {:.2f} {:.2f}\n'.format(
                CLASS_NAMES[final_class[u]].lower(), final_probs[u], final_boxes[u][0]-(final_boxes[u][2]/2), final_boxes[u][1]-(final_boxes[u][3]/2), final_boxes[u][2], final_boxes[u][3]) #L T w h
          )
        elif fmt == 'xyrb':
          f.write(
            '{:s} {:.6f} {:.2f} {:.2f} {:.2f} {:.2f}\n'.format(
                CLASS_NAMES[final_class[u]].lower(), final_probs[u], final_boxes[u][0]-(final_boxes[u][2]/2), final_boxes[u][1]-(final_boxes[u][3]/2), final_boxes[u][0]+(final_boxes[u][2]/2), final_boxes[u][1]+(final_boxes[u][3]/2)) #L T R B
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

          write_str = CLASS_NAMES[final_class[u]].lower()
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
    counter += 1
  print("Forward Pass Time", np.mean(np.asarray(time_values)*1000), "+/-", np.std(np.asarray(time_values)))
  print("Interpretation Time", np.mean(int_times), "+/-", np.std(int_times))
  print("NMS (softnms=", softnms, ") Time", np.mean(nms_times), "+/-", np.std(nms_times))
  print("Postprocessing Time", np.mean(post_times), "+/-", np.std(post_times))
  print("Total Time", np.mean(total_times), "+/-", np.std(total_times))
  with open(os.path.join(out_dir, 'runtime.txt'), 'w') as f:
      f.write(
          '{:s} {:.6f} {:s} {:.6f}\n'.format("Forward Pass Time: ", np.mean(np.asarray(time_values)*1000), " +/- ", np.std(np.asarray(time_values)))
      )
      f.write(
          '{:s} {:.6f} {:s} {:.6f}\n'.format("Interpretation Time: ", np.mean(int_times), "+/-", np.std(int_times))
      )
      f.write(
          '{:s} {:.6f} {:s} {:.6f}\n'.format("NMS Time: ", np.mean(nms_times), "+/-", np.std(nms_times))
      )
      f.write(
          '{:s} {:.6f} {:s} {:.6f}\n'.format("Postprocessing Time: ", np.mean(post_times), "+/-", np.std(post_times))
      )
      f.write(
          '{:s} {:.6f} {:s} {:.6f}\n'.format("Total Time: ", np.mean(total_times), "+/-", np.std(total_times))
      )
  f.close
  

def main(argv=None):
  if not tf.gfile.Exists(FLAGS.out_dir):
    tf.gfile.MakeDirs(FLAGS.out_dir)
  checkpoints = []
  if 'model.ckpt' in FLAGS.checkpoint:
    checkpoints.append(FLAGS.checkpoint)
  else:
    checkpoints.extend([os.path.join(x[0], 'frozen_inference_graph.pb') for x in os.walk(FLAGS.checkpoint) if 'inference' not in x[0] and 'frozen_inference_graph.pb' in x[2]])
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
      image_demo_inference_graph(FLAGS.label_path, mask_param, use_log_anchors, encoding_scheme, c, log_dir, frmt, softnms=False)
      # break

if __name__ == '__main__':
    tf.app.run()

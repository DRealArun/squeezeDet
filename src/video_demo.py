# Author: Arun Rajendra Prabhu (arun.rajendra.prabhu@iais.fraunhofer.de) 14/06/2020

"""SqueezeDet/SqueezeDetOcta video demo.

This implementation performs object segmentation on a sequence of images are a video
stream. The frames overlayed with the segmentation can be written to the disk if 
needed.
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
tf.app.flags.DEFINE_integer('mask_parameterization_now', 4,
                            """Bounding box is 4, octagonal mask is 8. other values not supported""")
tf.app.flags.DEFINE_boolean('log_anchors_now', False, """Use Log domain extracted anchors ?""")
tf.app.flags.DEFINE_string('encoding_type_now', 'normal',
                            """what type of encoding to use""")
tf.app.flags.DEFINE_boolean('write_to_disk', False, """Write the segmentations to disk ?""")

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
  overlap_matrix = np.zeros((len(order), len(order)))
  # count = 0
  for i in range(len(order)-1):
    ovps = batch_iou(boxes[order[i+1:]], boxes[order[i]])
    overlap_matrix[order[i], order[i+1:]] = ovps
    overlap_matrix[order[i+1:], order[i]] = ovps
    for j, ov in enumerate(ovps):
      # count +=1
      if ov > threshold:
        keep[order[j+i+1]] = False
        overlap_matrix[order[i], order[j+i+1]] = 0
        overlap_matrix[order[j+i+1], order[i]] = 0
  return keep, overlap_matrix

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
  max_overlaps = []
  NUM_CLASSES = 7
  if softnms:
      final_boxes = boxes
      final_probs = soft_nms(boxes, probs)
      final_cls_idx = cls_idx
  else:
      for c in range(NUM_CLASSES):
        idx_per_class = [i for i in range(len(probs)) if cls_idx[i] == c]
        keep, overlap_matrix = nms(boxes[idx_per_class], probs[idx_per_class], 0.5)
        for i in range(len(keep)):
          if keep[i]:
            final_boxes.append(boxes[idx_per_class[i]])
            final_probs.append(probs[idx_per_class[i]])
            final_cls_idx.append(c)
            max_overlaps.append(max(overlap_matrix[i,:]))
  return final_boxes, final_probs, final_cls_idx, max_overlaps

    
def interpret_output(output_volume, mask_parameterization=4, log_anchors=False, imgSize=(1024,512), encoding_type_now='normal'):
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
  # transforming back
  # if mask_parameterization == 8:
  width       = xmaxs - xmins
  height      = ymaxs - ymins
  # else:
  #   width       = xmaxs - xmins + 1.0
  #   height      = ymaxs - ymins + 1.0
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

def process_frame(img, mask_parameterization_now, log_anchors_now, encoding_type_now, checkpoint_path, 
                  sess, tensor_dict, image_tensor, softnms=False):
  IMAGE_WIDTH = 1024
  IMAGE_HEIGHT = 512
  PLOT_PROB_THRESH = 0.5
  PROB_THRESH = 0.005
  BGR_MEANS = np.array([[[103.939, 116.779, 123.68]]])
  image_np = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT))
  image_np_orig = copy.deepcopy(image_np)
  image_np = image_np.astype(np.float32, copy=False)
  image_unexpanded = image_np - BGR_MEANS
  image = np.expand_dims(image_unexpanded, axis=0)
  output_dict = sess.run(tensor_dict,
                         feed_dict={image_tensor: image})
  boxes, probs, classes = interpret_output(output_dict['conv12/bias_add'], mask_parameterization_now, log_anchors_now, [IMAGE_WIDTH, IMAGE_HEIGHT], encoding_type_now)
  det_bbox, det_prob, det_class, overlaps = filter_prediction(boxes, probs, classes, PLOT_PROB_THRESH, softnms)
  keep_idx    = [idx for idx in range(len(det_prob)) \
                    if det_prob[idx] >= 0.5]
  final_boxes = [det_bbox[idx] for idx in keep_idx]
  final_probs = [det_prob[idx] for idx in keep_idx]
  final_class = [det_class[idx] for idx in keep_idx]
  final_overlaps = [overlaps[idx] for idx in keep_idx]
  return final_boxes, final_probs, final_class, final_probs, image_np_orig

def video_demo_inference_graph(mask_parameterization_now, log_anchors_now, encoding_type_now, checkpoint_path, softnms=False):
  assert FLAGS.demo_net == 'squeezeDet', 'Selected neural net architecture not supported: {}'.format(FLAGS.demo_net)
  CLASS_NAMES = tuple(sorted(('person', 'rider', 'car', 'truck', 'bus', 'motorcycle', 'bicycle')))
  # Class specific color definitions
  _cdict = {}
  _cdict['person']      = (0,204,102)
  _cdict['rider']       = (102,0,204)
  _cdict['car']         = (0,000,204)
  _cdict['truck']       = (153,153,0)
  _cdict['bus']         = (0,153,153)
  _cdict['motorcycle']  = (204,0,102)
  _cdict['bicycle']     = (255,128,0)
  detection_graph = tf.Graph()
  if not '.png' in FLAGS.input_path:
    cap = cv2.VideoCapture(FLAGS.input_path)
    if (cap.isOpened()== False): 
      print("Error opening video stream or file")

  with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(checkpoint_path, 'rb') as fid:
      serialized_graph = fid.read()
      od_graph_def.ParseFromString(serialized_graph)
      tf.import_graph_def(od_graph_def, name='')
    with tf.Session() as sess:
      # Get handles to input and output tensors
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in ['conv12/bias_add']:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_input_1:0')
      if '.png' in FLAGS.input_path:
        print("Processing image sequences!")
        for image_path in glob.iglob(FLAGS.input_path):
          start = time.time()
          read_img = cv2.imread(image_path)
          frame_read_time = time.time()-start
          image_np_orig = copy.deepcopy(read_img)
          # Run inference
          final_boxes, final_probs, final_class, final_probs, image_np_orig = process_frame(read_img, mask_parameterization_now, \
                                      log_anchors_now, encoding_type_now, checkpoint_path, sess, tensor_dict, image_tensor, \
                                      softnms)
          end = time.time()
          fps = 1 / (end-start)
          fps_text = "FPS counter: {:0.2f} fps/Per frame read time: {:0.2f} ms".format(round(fps,2), round(frame_read_time*1000,2))
          _draw_box(
            image_np_orig, final_boxes,
            [CLASS_NAMES[idx]+': (%.2f)'% prob \
                for idx, prob in zip(final_class, final_probs)],
            (0, 0, 255), draw_masks=(mask_parameterization_now == 8), fill=False, cdict=_cdict, fps_text=fps_text)
          cv2.imshow('Frame',image_np_orig)
          if FLAGS.write_to_disk:
            out_file_name = os.path.join(FLAGS.out_dir, image_path.split('\\')[-1][:-4]+".png")
            cv2.imwrite(out_file_name, image_np_orig)
          if cv2.waitKey(25) & 0xFF == ord('q'):
            break
      else:
        frame_cnt = 0
        print("Processing video!")
        while(cap.isOpened()):
          start = time.time()
          ret, read_img = cap.read()
          frame_read_time = time.time()-start
          if ret == True:
            # Run inference
            final_boxes, final_probs, final_class, final_probs, image_np_orig = process_frame(read_img, mask_parameterization_now, \
                                        log_anchors_now, encoding_type_now, checkpoint_path, sess, tensor_dict, image_tensor, \
                                        softnms)
            end = time.time()
            fps = 1 / (end-start)
            fps_text = "FPS counter: {:0.2f} fps/Per frame read time: {:0.2f} ms".format(round(fps,2), round(frame_read_time*1000,2))
            _draw_box(
              image_np_orig, final_boxes,
              [CLASS_NAMES[idx]+': (%.2f)'% prob \
                  for idx, prob in zip(final_class, final_probs)],
              (0, 0, 255), draw_masks=(mask_parameterization_now == 8), fill=False, cdict=_cdict, fps_text=fps_text)
            cv2.imshow('Frame',image_np_orig)
            if FLAGS.write_to_disk:
              out_file_name = os.path.join(FLAGS.out_dir, FLAGS.input_path.split('\\')[-1][:-4]+"_"+str(frame_cnt)+".png")
              frame_cnt +=1
              cv2.imwrite(out_file_name, image_np_orig)
            if cv2.waitKey(25) & 0xFF == ord('q'):
              break
          else:
            break
  if not '.png' in FLAGS.input_path: 
    cap.release()
    cv2.destroyAllWindows()

def main(argv=None):
  video_demo_inference_graph(FLAGS.mask_parameterization_now, FLAGS.log_anchors_now, FLAGS.encoding_type_now, FLAGS.checkpoint, softnms=False)

if __name__ == '__main__':
  tf.app.run()
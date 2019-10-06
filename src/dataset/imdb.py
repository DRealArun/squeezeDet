# Author: Bichen Wu (bichen@berkeley.edu) 08/25/2016

"""The data base wrapper class"""

import os
import random
import shutil

from PIL import Image, ImageFont, ImageDraw
import cv2
import numpy as np
from utils.util import iou, batch_iou
import math
import collections

class Node_container(object):
    def __init__(self, connectivity_vector, id_val):
        self.connectivity_vector = connectivity_vector
        self.strength = -1
        self.id = id_val
        
    def set_strength(self, value):
        if value > self.strength:
            self.strength = value
    
    def get_strength(self):
        return self.strength
    
    def get_id(self):
        return self.id
    
    def get_child_ids(self):
        return np.where(self.connectivity_vector == 'bg')
    
def get_base_node_config(connectivity_matrix):
    base_node_ids = []
    for i, vector in enumerate(connectivity_matrix):
        unique, counts = np.unique(vector, return_counts=True)
        occurances = dict(zip(unique, counts))
        if '0' in occurances.keys():
            num_of_zeros = occurances['0']
        else:
            num_of_zeros = 0
        if 'fg' not in vector and num_of_zeros != len(vector)-1:
            base_node_ids.append(i)
    return base_node_ids

def get_end_node_config(connectivity_matrix):
    end_node_ids = []
    for i, vector in enumerate(connectivity_matrix):
        unique, counts = np.unique(vector, return_counts=True)
        occurances = dict(zip(unique, counts))
        if '0' in occurances.keys():
            num_of_zeros = occurances['0']
        else:
            num_of_zeros = 0
        if 'bg' not in vector and num_of_zeros != len(vector)-1:
            end_node_ids.append(i)
    return end_node_ids

def get_disconnected_node_config(connectivity_matrix):
    disconnected_node_ids = []
    for i, vector in enumerate(connectivity_matrix):
        unique, counts = np.unique(vector, return_counts=True)
        occurances = dict(zip(unique, counts))
        if '0' in occurances.keys():
            num_of_zeros = occurances['0']
        else:
            num_of_zeros = 0
        if num_of_zeros == len(vector)-1:
            disconnected_node_ids.append(i)
    return disconnected_node_ids

def print_network_dictionary(net_dictionary):
    for key in net_dictionary.keys():
        curr_node = net_dictionary[key]
        print("Strength of node ", key, " is ", curr_node.get_strength())
        
def sort_dict_by_strength(net_dictionary):
    sorted_nodes = sorted(net_dictionary.items(), key=lambda kv: kv[1].get_strength())
    sorted_dict = collections.OrderedDict(sorted_nodes)
    return sorted_dict

def resolve_node_strengths(connectivity_matrix):
    # print(connectivity_matrix)
    num_nodes = len(connectivity_matrix)
    base_node_ids = get_base_node_config(connectivity_matrix)
    num_base_nodes = len(base_node_ids)
    # print("Number of base nodes:", num_base_nodes, base_node_ids)
    end_node_ids = get_end_node_config(connectivity_matrix)
    num_end_nodes = len(end_node_ids)
    # print("Number of end nodes:", num_end_nodes, end_node_ids)
    disconnected_node_ids = get_disconnected_node_config(connectivity_matrix)
    num_disconnected_nodes = len(disconnected_node_ids)
    # print("Number of disconnected nodes:", num_disconnected_nodes, disconnected_node_ids)

    node_list = []
    for base_id in base_node_ids:
        base_node = Node_container(connectivity_matrix[base_id], base_id)
        base_node.set_strength(0.)
        node_list.append(base_node)

    for disconnect_id in disconnected_node_ids:
        disconnected_node = Node_container(connectivity_matrix[disconnect_id], disconnect_id)
        disconnected_node.set_strength(1.0)
        node_list.append(disconnected_node)

    traversed_dict = {}
    if (num_base_nodes+num_disconnected_nodes) != num_nodes: #if this is true then it implies that num_base_nodes and num_end_nodes = 0, hence all nodes are disconnected
      strength_step = 1.0/(num_nodes-num_base_nodes-num_disconnected_nodes)
      # print("Strength_increment:",strength_step)
      while len(node_list) != 0:
          curr_node = node_list.pop(0)
          traversed_dict[curr_node.get_id()] = curr_node
          child_ids = curr_node.get_child_ids()[0]
          for child_id in child_ids:
              if child_id not in traversed_dict:
                  child_node = Node_container(connectivity_matrix[child_id], child_id)
                  if child_id not in end_node_ids:
                      child_node.set_strength(curr_node.get_strength()+strength_step)
                  else:
                      child_node.set_strength(1.0)
                  node_list.append(child_node)
              else:
                  if child_id not in end_node_ids:
                      child_node = traversed_dict[child_id]
                      child_node.set_strength(curr_node.get_strength()+strength_step)
    else:
      # Since all nodes are disconnected, simply add them to the dictionary
      while len(node_list) != 0:
          curr_node = node_list.pop(0)
          traversed_dict[curr_node.get_id()] = curr_node


    return traversed_dict

def batch_iou_mask(masks, mask):
    """Compute the Intersection-Over-Union of a batch of masks with another
    mask.

    Args:
    masks: batch of binary masks
    mask: a single array mask
    Returns:
    ious: array of a float number in range [0, 1].
    """
    num_masks = np.shape(masks)
    masks_reshaped = np.reshape(masks, (num_masks[0], num_masks[1]*num_masks[2]))
    mask_reshaped = np.reshape(mask, (num_masks[1]*num_masks[2]))
#     print(np.shape(np.bitwise_and(masks_reshaped, mask_reshaped)))
    inter = np.sum(np.bitwise_and(masks_reshaped, mask_reshaped), axis=1)
    union = np.sum(np.bitwise_or(masks_reshaped, mask_reshaped), axis=1)
    if math.nan in union or math.inf in union or math.nan in inter or math.inf in inter:
        print("IOU", inter, union)
    return inter/union

def find_overlaps_masks(id_val, object_list, exclude_mask):
    # Takes complement of exclude mask and then only tries to find overlap with the included objects.
    # Returns the overlapping objects with it.
    search_space = object_list #need to handle excluded mask
#     print(search_space)
    candidate_mask = object_list[id_val]
    overlaps = batch_iou_mask(search_space, candidate_mask)
    indices = np.where(overlaps > 0.001)[0]
#     print(id_val, overlaps)
    return indices

class imdb(object):
  """Image database."""

  def __init__(self, name, mc):
    self._name = name
    self._classes = []
    self._image_set = []
    self._image_idx = []
    self._data_root_path = []
    self._rois = {}
    self._poly = {}
    self.mc = mc

    # batch reader
    self._perm_idx = None
    self._cur_idx = 0

  @property
  def name(self):
    return self._name

  @property
  def classes(self):
    return self._classes

  @property
  def num_classes(self):
    return len(self._classes)

  @property
  def image_idx(self):
    return self._image_idx

  @property
  def image_set(self):
    return self._image_set

  @property
  def data_root_path(self):
    return self._data_root_path

  @property
  def year(self):
    return self._year

  def _shuffle_image_idx(self):
    self._perm_idx = [self._image_idx[i] for i in
        np.random.permutation(np.arange(len(self._image_idx)))]
    self._cur_idx = 0

  def read_image_batch(self, shuffle=True):
    """Only Read a batch of images
    Args:
      shuffle: whether or not to shuffle the dataset
    Returns:
      images: length batch_size list of arrays [height, width, 3]
    """
    mc = self.mc
    if shuffle:
      if self._cur_idx + mc.BATCH_SIZE >= len(self._image_idx):
        self._shuffle_image_idx()
      batch_idx = self._perm_idx[self._cur_idx:self._cur_idx+mc.BATCH_SIZE]
      self._cur_idx += mc.BATCH_SIZE
    else:
      if self._cur_idx + mc.BATCH_SIZE >= len(self._image_idx):
        batch_idx = self._image_idx[self._cur_idx:] \
            + self._image_idx[:self._cur_idx + mc.BATCH_SIZE-len(self._image_idx)]
        self._cur_idx += mc.BATCH_SIZE - len(self._image_idx)
      else:
        batch_idx = self._image_idx[self._cur_idx:self._cur_idx+mc.BATCH_SIZE]
        self._cur_idx += mc.BATCH_SIZE

    images, scales = [], []
    for i in batch_idx:
      im = cv2.imread(self._image_path_at(i))
      im = im.astype(np.float32, copy=False)
      im -= mc.BGR_MEANS
      orig_h, orig_w, _ = [float(v) for v in im.shape]
      im = cv2.resize(im, (mc.IMAGE_WIDTH, mc.IMAGE_HEIGHT))
      x_scale = mc.IMAGE_WIDTH/orig_w
      y_scale = mc.IMAGE_HEIGHT/orig_h
      images.append(im)
      scales.append((x_scale, y_scale))

    return images, scales

  def get_perpendicular_distance(self, pt1, m, pt2):
    pt1_x, pt1_y = pt1 #line
    pt2_x, pt2_y = pt2 #point
    c = pt1_y - (m*pt1_x)
    # Ax+By+C=0 -> y=mx+c -> y-mx-c=0 -> -mx+y-c=0 -> A = -m, B=1, C=-c
    # perpendicular distance = |A*x2 + B*y2 + C| /sqrt((A**2+B**2))
    A = -m
    B = 1
    C = -c
    offset = abs((A*pt2_x + B*pt2_y + C))/((A**2+B**2)**(0.5))
    return offset

  def get_intersecting_point_new(self, vert_hor, eq1, pt, m):
      pt_x, pt_y = pt
      c = pt_y - (m*pt_x)
      if vert_hor == "vert":
          x_cor = eq1
          y_cor = (m*x_cor) + c
      else:
          y_cor = eq1
          x_cor = (y_cor - c)/m
      return (x_cor, y_cor)

  def decode_parameterization(self, mask_vector):
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
    m1 = (pts[2][1]-pts[0][1])/(pts[2][0]-pts[0][0])
    m2 = (pts[3][1]-pts[1][1])/(pts[3][0]-pts[1][0])
    ms = [-1/m1, -1/m2, -1/m2, -1/m1, -1/m1, -1/m2, -1/m2, -1/m1]
    intersecting_pts = []
    for eq1, pt, vert_hor, m in zip(eq1s, points, vert_or_hors, ms):
        op_pt = self.get_intersecting_point_new(vert_hor, eq1, pt, m)
        intersecting_pts.append(op_pt)
    return intersecting_pts

  def get_8_point_mask(self, polygon, h, w):
    outline = np.array(polygon)
    rr, cc = outline[:,1], outline[:,0]
    # rrr, ccc = outline[:,1], outline[:,0]
    # rr = []
    # cc = []
    # for r in rrr:
    #   if r < 0:
    #     r = 0
    #   if r > h:
    #     r = h
    #   rr.append(r)
    # for c in ccc:
    #   if c < 0:
    #     c = 0
    #   if c > w:
    #     c = w
    #   cc.append(c)
    rr = np.array(rr)
    cc = np.array(cc)
    sum_values = cc + rr
    diff_values = cc - rr
    xmin = max(min(cc), 0)
    xmax = min(max(cc), w)
    ymin = max(min(rr), 0)
    ymax = min(max(rr), h)
    width       = xmax - xmin
    height      = ymax - ymin
    if width <= 0:
      print("Max and min x values", xmax, xmin, w)
    if height <= 0:
      print("Max and min y values", ymax, ymin, h)
    center_x  = xmin + 0.5*width 
    center_y  = ymin + 0.5*height
    center = (center_x, center_y)
    min_sum_indices = np.where(sum_values == np.amin(sum_values))[0][0]
    pt_p_min = (cc[min_sum_indices], rr[min_sum_indices])
    max_sum_indices = np.where(sum_values == np.amax(sum_values))[0][0]
    pt_p_max = (cc[max_sum_indices], rr[max_sum_indices])
    min_diff_indices = np.where(diff_values == np.amin(diff_values))[0][0]
    pt_n_min = (cc[min_diff_indices], rr[min_diff_indices])
    max_diff_indices = np.where(diff_values == np.amax(diff_values))[0][0]
    pt_n_max = (cc[max_diff_indices], rr[max_diff_indices])
    pts = [pt_p_min, pt_n_min, pt_p_max, pt_n_max]
    ms = [-1, +1, -1,  +1] #Slope of the tangents
    offsets = []
    for pt, m in zip(pts, ms):
      op_pt = self.get_perpendicular_distance(pt, m, center)
      offsets.append(op_pt)
    mask_vector = [center_x, center_y, width, height, offsets[0], offsets[1], offsets[2], offsets[3]]
    return mask_vector

  def read_batch(self, shuffle=True):
    """Read a batch of image and bounding box annotations.
    Args:
      shuffle: whether or not to shuffle the dataset
    Returns:
      image_per_batch: images. Shape: batch_size x width x height x [b, g, r]
      label_per_batch: labels. Shape: batch_size x object_num
      delta_per_batch: bounding box deltas. Shape: batch_size x object_num x 
          [dx ,dy, dw, dh]
      aidx_per_batch: index of anchors that are responsible for prediction.
          Shape: batch_size x object_num
      bbox_per_batch: scaled bounding boxes. Shape: batch_size x object_num x 
          [cx, cy, w, h]
    """
    mc = self.mc

    if shuffle:
      if self._cur_idx + mc.BATCH_SIZE >= len(self._image_idx):
        self._shuffle_image_idx()
      batch_idx = self._perm_idx[self._cur_idx:self._cur_idx+mc.BATCH_SIZE]
      self._cur_idx += mc.BATCH_SIZE
    else:
      if self._cur_idx + mc.BATCH_SIZE >= len(self._image_idx):
        batch_idx = self._image_idx[self._cur_idx:] \
            + self._image_idx[:self._cur_idx + mc.BATCH_SIZE-len(self._image_idx)]
        self._cur_idx += mc.BATCH_SIZE - len(self._image_idx)
      else:
        batch_idx = self._image_idx[self._cur_idx:self._cur_idx+mc.BATCH_SIZE]
        self._cur_idx += mc.BATCH_SIZE

    image_per_batch = []
    label_per_batch = []
    bbox_per_batch  = []
    delta_per_batch = []
    aidx_per_batch  = []
    strength_per_batch = []
    if mc.DEBUG_MODE:
      avg_ious = 0.
      num_objects = 0.
      max_iou = 0.0
      min_iou = 1.0
      num_zero_iou_obj = 0

    for img_ct, idx in enumerate(batch_idx):
      # load the image
      try:
        Image.open(self._image_path_at(idx)).tobytes()
      except IOError:
        print('Detect error img %s' % self._image_path_at(idx))
        continue
      im = cv2.imread(self._image_path_at(idx))
      if im is None:
        print("\n\nCorrupt image found: ", self._image_path_at(idx))
        continue

      im = im.astype(np.float32, copy=False)
      im -= mc.BGR_MEANS
      orig_h, orig_w, _ = [float(v) for v in im.shape]

      # load annotations
      label_per_batch.append([b[4] for b in self._rois[idx][:]])
      # gt_bbox = np.array([[b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]] for b in self._rois[idx][:]])
      gt_bbox_pre = np.array([[b[0], b[1], b[2], b[3]] for b in self._rois[idx][:]])
      polygons = [b[2] for b in self._poly[idx][:]]
      flag1 = False
      flag2 = False
      if mc.DATA_AUGMENTATION:
        assert mc.DRIFT_X >= 0 and mc.DRIFT_Y > 0, \
            'mc.DRIFT_X and mc.DRIFT_Y must be >= 0'

        if mc.DRIFT_X > 0 or mc.DRIFT_Y > 0:
          # Ensures that gt boundibg box is not cutted out of the image
          max_drift_x = min(gt_bbox_pre[:, 0] - gt_bbox_pre[:, 2]/2.0+1)
          max_drift_y = min(gt_bbox_pre[:, 1] - gt_bbox_pre[:, 3]/2.0+1)
          assert max_drift_x >= 0 and max_drift_y >= 0, 'bbox out of image'

          dy = np.random.randint(-mc.DRIFT_Y, min(mc.DRIFT_Y+1, max_drift_y))
          dx = np.random.randint(-mc.DRIFT_X, min(mc.DRIFT_X+1, max_drift_x))

          # shift bbox
          gt_bbox_pre[:, 0] = gt_bbox_pre[:, 0] - dx
          gt_bbox_pre[:, 1] = gt_bbox_pre[:, 1] - dy
          flag1 = True
          # distort image
          orig_h -= dy
          orig_w -= dx
          orig_x, dist_x = max(dx, 0), max(-dx, 0)
          orig_y, dist_y = max(dy, 0), max(-dy, 0)

          distorted_im = np.zeros(
              (int(orig_h), int(orig_w), 3)).astype(np.float32)
          distorted_im[dist_y:, dist_x:, :] = im[orig_y:, orig_x:, :]
          im = distorted_im

        # # Flip image with 50% probability
        if np.random.randint(2) > 0.5:
          im = im[:, ::-1, :]
          flag2 = True
          gt_bbox_pre[:, 0] = orig_w - 1 - gt_bbox_pre[:, 0]
          # gt_bbox_pre[:, 4] = orig_w - 1 - gt_bbox_pre[:, 4]
          # gt_bbox_pre[:, 6] = orig_w - 1 - gt_bbox_pre[:, 6]
          # gt_bbox_pre[:, 8] = orig_w - 1 - gt_bbox_pre[:, 8]
          # gt_bbox_pre[:, 10] = orig_w - 1 - gt_bbox_pre[:, 10]

      temp = list(polygons)
      # scale image
      im = cv2.resize(im, (mc.IMAGE_WIDTH, mc.IMAGE_HEIGHT))
      image_per_batch.append(im)

      # scale annotation
      x_scale = mc.IMAGE_WIDTH/orig_w
      y_scale = mc.IMAGE_HEIGHT/orig_h
      gt_bbox_pre[:, 0:4:2] = (gt_bbox_pre[:, 0:4:2]*x_scale)
      gt_bbox_pre[:, 1:4:2] = (gt_bbox_pre[:, 1:4:2]*y_scale)
      for m in range(len(polygons)):
        poly = np.array(polygons[m])
        if flag1:
          poly[:,0] = poly[:,0] - dx
          poly[:,1] = poly[:,1] - dy
        if flag2:
          poly[:,0] = orig_w - 1 - poly[:,0]
        poly[:,0] = poly[:,0]*x_scale
        poly[:,1] = poly[:,1]*y_scale
        polygons[m] = poly
      flag1 = False
      flag2 = False
      # gt_bbox[:, 6:8:1] = (np.round(gt_bbox[:, 6:8:1]*x_scale)).astype(np.int)
      # gt_bbox[:, 8:10:1] = (np.round(gt_bbox[:, 8:10:1]*y_scale)).astype(np.int)
      # gt_bbox[:, 10:12:1] =(np.round(gt_bbox[:, 10:12:1]*x_scale)).astype(np.int)

      # Transform the bounding box to offset mode
      gt_bbox = []
      actual_bin_masks = []
      for o in range(len(polygons)):
        polygon = polygons[o]
        mask_vector = self.get_8_point_mask(polygon, mc.IMAGE_HEIGHT, mc.IMAGE_WIDTH)
        center_x, center_y, width, height, of1, of2, of3, of4 = mask_vector
        if width == 0 or height == 0:
          print("Error width and height", width, height, gt_bbox_pre[o][2], gt_bbox_pre[o][3], center_x, center_y, gt_bbox_pre[o][0], gt_bbox_pre[o][1], idx)
          # del label_per_batch[img_ct][o]
          # continue #ONLY FOR COMPLETE SET OF MASKS
        assert not (of1 < 0 or of2 < 0 or of3 < 0 or of4 < 0), "Error Occured "+ str(of1) +" "+ str(of2)+" "+ str(of3)+" "+ str(of4)
        bbox = mask_vector
        points = self.decode_parameterization(bbox)
        points = np.round(points)
        points = np.array(points, 'int32')
        assert not (points[0][1] - points[1][1] > 1 or points[2][0] - points[3][0] > 1 or points[5][1] - points[4][1] > 1 or points[7][0] - points[6][0] > 1), "\n\n Error in extraction:"+str(points)+" "+str(idx)+" "+str(bbox)
        gt_bbox.append(bbox)

        drawing2 = np.zeros((mc.IMAGE_HEIGHT, mc.IMAGE_WIDTH, 3), np.uint8)
        color = (255, 255, 255)
        cv2.fillConvexPoly(drawing2, np.array(polygon, 'int32'), color)
        gray2 = cv2.cvtColor(drawing2, cv2.COLOR_BGR2GRAY) # convert to grayscale
        refined_mask_2 = gray2 / 255
        refined_mask_2 = refined_mask_2.astype(np.uint8)
        actual_bin_masks.append(refined_mask_2)

      bbox_per_batch.append(gt_bbox)

      num_objects = len(gt_bbox)
      ct_matrix = np.ones((num_objects,num_objects)) * -1
      ct_matrix = ct_matrix.astype(str)
      np.fill_diagonal(ct_matrix, 'x')
      for m in range(num_objects):
  #         overlapping_objects = find_overlaps(i, np.array(box_proposals), np.zeros((len(box_proposals),), dtype=int)) # get a overlap contention mask which determines which objects need to be considered while finding the overlaps
          overlapping_objects = find_overlaps_masks(m, np.array(actual_bin_masks), np.zeros((num_objects,), dtype=int)) # get a overlap contention mask which determines which objects need to be considered while finding the overlaps
          for n in overlapping_objects:
              if ct_matrix[m][n] == '-1.0':
                  ct_matrix[m][n] = 'bg'
                  ct_matrix[n][m] = 'fg'
      refined_matrix = np.where(ct_matrix=='-1.0', 0, ct_matrix)
      traversed_dictionary = resolve_node_strengths(refined_matrix)
      sorted_dict = sort_dict_by_strength(traversed_dictionary)

      aidx_per_image, delta_per_image, strength_per_image = [], [], []
      aidx_set = set()
      for i in range(len(gt_bbox)):
        encompassing_box = gt_bbox[i][0:4]
        overlaps = batch_iou(mc.ANCHOR_BOX, encompassing_box)
        strength = sorted_dict[i].get_strength()
        aidx = len(mc.ANCHOR_BOX)
        for ov_idx in np.argsort(overlaps)[::-1]:
          if overlaps[ov_idx] <= 0:
            if mc.DEBUG_MODE:
              min_iou = min(overlaps[ov_idx], min_iou)
              num_objects += 1
              num_zero_iou_obj += 1
            break
          if ov_idx not in aidx_set:
            aidx_set.add(ov_idx)
            aidx = ov_idx
            if mc.DEBUG_MODE:
              max_iou = max(overlaps[ov_idx], max_iou)
              min_iou = min(overlaps[ov_idx], min_iou)
              avg_ious += overlaps[ov_idx]
              num_objects += 1
            break

        if aidx == len(mc.ANCHOR_BOX): 
          # even the largeset available overlap is 0, thus, choose one with the
          # smallest square distance
          dist = np.sum(np.square(encompassing_box - mc.ANCHOR_BOX), axis=1)
          for dist_idx in np.argsort(dist):
            if dist_idx not in aidx_set:
              aidx_set.add(dist_idx)
              aidx = dist_idx
              break

        box_cx, box_cy, box_w, box_h, of1, of2, of3, of4 = gt_bbox[i]
        delta = [0]*8
        EPSILON = 1e-8
        delta[0] = (box_cx - mc.ANCHOR_BOX[aidx][0])/mc.ANCHOR_BOX[aidx][2]
        delta[1] = (box_cy - mc.ANCHOR_BOX[aidx][1])/mc.ANCHOR_BOX[aidx][3]
        delta[2] = np.log(box_w/mc.ANCHOR_BOX[aidx][2])
        delta[3] = np.log(box_h/mc.ANCHOR_BOX[aidx][3])

        anchor_diagonal = (mc.ANCHOR_BOX[aidx][2]**2+mc.ANCHOR_BOX[aidx][3]**2)**(0.5)
        delta[4] = np.log((of1 + EPSILON)/anchor_diagonal)
        delta[5] = np.log((of2 + EPSILON)/anchor_diagonal)

        delta[6] = np.log((of3 + EPSILON)/anchor_diagonal)
        delta[7] = np.log((of4 + EPSILON)/anchor_diagonal)

        aidx_per_image.append(aidx)
        delta_per_image.append(delta)
        strength_per_image.append(strength)

      delta_per_batch.append(delta_per_image)
      aidx_per_batch.append(aidx_per_image)
      strength_per_batch.append(strength_per_image)

    if mc.DEBUG_MODE:
      print ('max iou: {}'.format(max_iou))
      print ('min iou: {}'.format(min_iou))
      print ('avg iou: {}'.format(avg_ious/num_objects))
      print ('number of objects: {}'.format(num_objects))
      print ('number of objects with 0 iou: {}'.format(num_zero_iou_obj))

    return image_per_batch, label_per_batch, delta_per_batch, \
        aidx_per_batch, bbox_per_batch, strength_per_batch

  def evaluate_detections(self):
    raise NotImplementedError

  def visualize_detections(
      self, image_dir, image_format, det_error_file, output_image_dir,
      num_det_per_type=10):

    # load detections
    with open(det_error_file) as f:
      lines = f.readlines()
      random.shuffle(lines)
    f.close()

    dets_per_type = {}
    for line in lines:
      obj = line.strip().split(' ')
      error_type = obj[1]
      if error_type not in dets_per_type:
        dets_per_type[error_type] = [{
            'im_idx':obj[0], 
            'bbox':[float(obj[2]), float(obj[3]), float(obj[4]), float(obj[5])],
            'class':obj[6],
            'score': float(obj[7])
        }]
      else:
        dets_per_type[error_type].append({
            'im_idx':obj[0], 
            'bbox':[float(obj[2]), float(obj[3]), float(obj[4]), float(obj[5])],
            'class':obj[6],
            'score': float(obj[7])
        })

    out_ims = []
    # Randomly select some detections and plot them
    COLOR = (200, 200, 0)
    for error_type, dets in dets_per_type.iteritems():
      det_im_dir = os.path.join(output_image_dir, error_type)
      if os.path.exists(det_im_dir):
        shutil.rmtree(det_im_dir)
      os.makedirs(det_im_dir)

      for i in range(min(num_det_per_type, len(dets))):
        det = dets[i]
        im = Image.open(
            os.path.join(image_dir, det['im_idx']+image_format))
        draw = ImageDraw.Draw(im)
        draw.rectangle(det['bbox'], outline=COLOR)
        draw.text((det['bbox'][0], det['bbox'][1]), 
                  '{:s} ({:.2f})'.format(det['class'], det['score']),
                  fill=COLOR)
        out_im_path = os.path.join(det_im_dir, str(i)+image_format)
        im.save(out_im_path)
        im = np.array(im)
        out_ims.append(im[:,:,::-1]) # RGB to BGR
    return out_ims


# Author: Arun Prabhu (arun.rajendra.prabhu@iais.fraunhofer.de) 10/11/2019

"""The data base wrapper class around imdb to override the read_batch function"""

import os
import random
import shutil
import math

from dataset.imdb import imdb
from PIL import Image, ImageFont, ImageDraw
import cv2
import copy
import numpy as np
from utils.util import iou, batch_iou, bbox_transform

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
  for eq1, pt, vert_hor, m in zip(eq1s, points, vert_or_hors, ms):
      op_pt = _get_intersecting_point(vert_hor, eq1, pt, m)
      intersecting_pts.append(op_pt)
  return intersecting_pts

class input_reader(imdb):
  """Base image database handler for instance based annotations."""

  def __init__(self, name, mc):
    imdb.__init__(self, name, mc)
    self.left_margin = 0
    self.right_margin = 0
    self.top_margin = 0
    self.bottom_margin = 0


  def _get_perpendicular_distance(self, pt1, m, pt2):
    """Finds the perpendicular distance between a point and a line.
    Args:
      pt1: point on the line
      m  : slope of the line
      pt2: point from which perpendicular distance to line is to be found.
    Returns:
      offset: perpendicular distance between line and the point.
    """
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

  def _get_8_point_mask(self, polygon, h, w):
    """Finds the safe octagonal encoding of the polygon.
    Args:
      polygon: list of points representing the mask
      h      : height of the image
      w      : width of the image
    Returns:
      mask vector: [cx, cy, w, h, of1, of2, of3, of4]
    """
    outline = np.array(polygon)
    rrr, ccc = outline[:,1], outline[:,0]
    rr = []
    cc = []
    for r in rrr:
      if r < 0:
        r = 0
      if r > h:
        r = h
      rr.append(r)
    for c in ccc:
      if c < 0:
        c = 0
      if c > w:
        c = w
      cc.append(c)
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
      op_pt = self._get_perpendicular_distance(pt, m, center)
      offsets.append(op_pt)
    mask_vector = [center_x, center_y, width, height, offsets[0], offsets[1], offsets[2], offsets[3]]
    return mask_vector

  def read_batch(self, shuffle=True, wrap_around=True):
    """Read a batch of image and instance annotations.
    Args:
      shuffle: whether or not to shuffle the dataset
      wrap_around: cyclic data extraction
    Returns:
      image_per_batch: images. Shape: batch_size x width x height x [b, g, r]
      label_per_batch: labels. Shape: batch_size x object_num
      delta_per_batch: bounding box or mask deltas. Shape: batch_size x object_num x 
          [dx ,dy, dw, dh] or [dx, dy, dw, dh, dof1, dof2, dof3, dof4]
      aidx_per_batch: index of anchors that are responsible for prediction.
          Shape: batch_size x object_num
      bbox_per_batch: scaled bounding boxes or mask parameters. Shape: batch_size x object_num x 
          [cx, cy, w, h] or [cx, cy, w, h, of1, of2, of3, of4]
    """
    mc = self.mc

    if shuffle:
      if self._cur_idx + mc.BATCH_SIZE >= len(self._image_idx):
        self._shuffle_image_idx()
      batch_idx = self._perm_idx[self._cur_idx:self._cur_idx+mc.BATCH_SIZE]
      self._cur_idx += mc.BATCH_SIZE
    else:
      # Check for warp around only in non shuffle mode
      if self._cur_idx + mc.BATCH_SIZE >= len(self._image_idx):
        batch_idx = self._image_idx[self._cur_idx:] \
            + self._image_idx[:self._cur_idx + mc.BATCH_SIZE-len(self._image_idx)]
        if wrap_around:
          self._cur_idx += mc.BATCH_SIZE - len(self._image_idx)
        else:
          # Restart the counter if no-wrap-around is enabled
          # This ensures all the validation examples are evaluated
          self._cur_idx = 0
      else:
        batch_idx = self._image_idx[self._cur_idx:self._cur_idx+mc.BATCH_SIZE]
        self._cur_idx += mc.BATCH_SIZE

    image_per_batch = []
    label_per_batch = []
    bbox_per_batch  = []
    delta_per_batch = []
    aidx_per_batch  = []
    boundary_adhesions_per_batch = []
    if mc.DEBUG_MODE:
      avg_ious = 0.
      num_objects = 0.
      max_iou = 0.0
      min_iou = 1.0
      num_zero_iou_obj = 0

    for img_ct, idx in enumerate(batch_idx):
      # load the image
      try:
        # Seems to be the only way to detect invalid image files
        Image.open(self._image_path_at(idx)).tobytes()
      except IOError:
        print('Detect error img %s' % self._image_path_at(idx))
        continue
      im = cv2.imread(self._image_path_at(idx)).astype(np.float32, copy=False)
      if im is None:
        print("\n\nCorrupt image found: ", self._image_path_at(idx))
        continue

      im = im.astype(np.float32, copy=False)
      im -= mc.BGR_MEANS
      orig_h, orig_w, _ = [float(v) for v in im.shape]

      # load annotations
      label_per_batch.append([b[4] for b in self._rois[idx][:]])
      gt_bbox_pre = np.array([[b[0], b[1], b[2], b[3]] for b in self._rois[idx][:]])
      boundary_adhesion_pre = np.array([[b[0], b[1], b[2], b[3]] for b in self._boundary_adhesions[idx][:]])

      if mc.EIGHT_POINT_REGRESSION:
        polygons = [b[2] for b in self._poly[idx][:]]
      is_drift_performed = False
      is_flip_performed = False

      assert np.all((gt_bbox_pre[:, 0] - (gt_bbox_pre[:, 2]/2.0)) >= 0) or \
              np.all((gt_bbox_pre[:, 0] + (gt_bbox_pre[:, 2]/2.0)) < orig_w), "Error in the bounding boxes befire augmentation"

      if mc.DATA_AUGMENTATION:
        assert mc.DRIFT_X >= 0 and mc.DRIFT_Y > 0, \
            'mc.DRIFT_X and mc.DRIFT_Y must be >= 0'

        if mc.DRIFT_X > 0 or mc.DRIFT_Y > 0:
          # Ensures that gt bounding box is not cut out of the image
          max_drift_x = math.floor(min(gt_bbox_pre[:, 0] - (gt_bbox_pre[:, 2]/2.0)+1))
          max_drift_y = math.floor(min(gt_bbox_pre[:, 1] - (gt_bbox_pre[:, 3]/2.0)+1))
          assert max_drift_x >= 0 and max_drift_y >= 0, 'bbox out of image'

          dy = np.random.randint(-mc.DRIFT_Y, min(mc.DRIFT_Y+1, max_drift_y))
          dx = np.random.randint(-mc.DRIFT_X, min(mc.DRIFT_X+1, max_drift_x))

          # shift bbox
          gt_bbox_pre[:, 0] = gt_bbox_pre[:, 0] - dx
          gt_bbox_pre[:, 1] = gt_bbox_pre[:, 1] - dy
          is_drift_performed = True
          # distort image
          orig_h -= dy
          orig_w -= dx
          orig_x, dist_x = max(dx, 0), max(-dx, 0)
          orig_y, dist_y = max(dy, 0), max(-dy, 0)

          distorted_im = np.zeros(
              (int(orig_h), int(orig_w), 3)).astype(np.float32)
          distorted_im[dist_y:, dist_x:, :] = im[orig_y:, orig_x:, :]
          dist_h, dist_w, _ = [float(v) for v in distorted_im.shape]
          im = distorted_im

          if dx < 0:
            # Recheck right boundary
            xmax_temp = gt_bbox_pre[:, 0] + (gt_bbox_pre[:, 2]/2)
            temp_ids = np.where(xmax_temp >= dist_w-1-self.right_margin)[0]
            if len(temp_ids) > 0:
              boundary_adhesion_pre[temp_ids,2] = True # Right boundary
          if dy < 0:
            # Recheck bottom boundary
            ymax_temp = gt_bbox_pre[:, 1] + (gt_bbox_pre[:, 3]/2)
            temp_ids = np.where(ymax_temp >= dist_h-1-self.bottom_margin)[0]
            if len(temp_ids) > 0:
              boundary_adhesion_pre[temp_ids,3] = True # Bottom boundary
          if dx > 0:
            # Recheck left boundary
            xmin_temp = gt_bbox_pre[:, 0] - (gt_bbox_pre[:, 2]/2)
            temp_ids = np.where(xmin_temp <= self.left_margin)[0]
            if len(temp_ids) > 0:
              boundary_adhesion_pre[temp_ids,0] = True # Left boundary
          if dy > 0:
            # Recheck top boundary
            ymin_temp = gt_bbox_pre[:, 1] - (gt_bbox_pre[:, 3]/2)
            temp_ids = np.where(ymin_temp <= self.top_margin)[0]
            if len(temp_ids) > 0:
              boundary_adhesion_pre[temp_ids,1] = True # Top boundary


        # Flip image with 50% probability
        if np.random.randint(2) > 0.5:
          im = im[:, ::-1, :]
          is_flip_performed = True
          gt_bbox_pre[:, 0] = orig_w - 1 - gt_bbox_pre[:, 0]
          temp = copy.deepcopy(boundary_adhesion_pre[:,0])
          boundary_adhesion_pre[:,0] = boundary_adhesion_pre[:,2]
          boundary_adhesion_pre[:,2] = temp 

      # scale image
      im = cv2.resize(im, (mc.IMAGE_WIDTH, mc.IMAGE_HEIGHT))
      image_per_batch.append(im)

      # scale annotation
      x_scale = mc.IMAGE_WIDTH/orig_w
      y_scale = mc.IMAGE_HEIGHT/orig_h
      gt_bbox_pre[:, 0::2] = gt_bbox_pre[:, 0::2]*x_scale
      gt_bbox_pre[:, 1::2] = gt_bbox_pre[:, 1::2]*y_scale

      assert np.all((gt_bbox_pre[:, 0] - (gt_bbox_pre[:, 2]/2.0)) >= 0) or \
              np.all((gt_bbox_pre[:, 0] + (gt_bbox_pre[:, 2]/2.0)) < orig_w), "Error in the bounding boxes after augmentation"
      if mc.EIGHT_POINT_REGRESSION:
        for p in range(len(polygons)):
          poly = np.array(polygons[p])
          if is_drift_performed:
            poly[:,0] = poly[:,0] - dx
            poly[:,1] = poly[:,1] - dy
          if is_flip_performed:
            poly[:,0] = orig_w - 1 - poly[:,0]
          poly[:,0] = poly[:,0]*x_scale
          poly[:,1] = poly[:,1]*y_scale
          polygons[p] = poly
      is_drift_performed = False
      is_flip_performed = False
      gt_bbox = gt_bbox_pre # Use shifted bounding box if EIGHT_POINT_REGRESSION = False
      # Transform the bounding box to offset mode.
      # We extract the bounding box from the flipped and drifted masks to ensure
      # consistency.
      if mc.EIGHT_POINT_REGRESSION:
        gt_bbox = []
        actual_bin_masks = []
        for k in range(len(polygons)):
          polygon = polygons[k]
          mask_vector = self._get_8_point_mask(polygon, mc.IMAGE_HEIGHT, mc.IMAGE_WIDTH)
          center_x, center_y, width, height, of1, of2, of3, of4 = mask_vector
          if width == 0 or height == 0:
            print("Error in width or height", width, height, gt_bbox_pre[k][2], gt_bbox_pre[k][3], center_x, center_y, gt_bbox_pre[k][0], gt_bbox_pre[k][1], idx)
            del label_per_batch[img_ct][k]
            continue
          assert not (of1 <= 0 or of2 <= 0 or of3 <= 0 or of4 <= 0), "Error Occured "+ str(of1) +" "+ str(of2)+" "+ str(of3)+" "+ str(of4)
          points = decode_parameterization(mask_vector)
          points = np.round(points)
          points = np.array(points, 'int32')
          assert not ((points[0][1] - points[1][1]) > 1 or (points[2][0] - points[3][0]) > 1 or (points[5][1] - points[4][1]) > 1 or (points[7][0] - points[6][0]) > 1), \
            "\n\n Error in extraction:"+str(points)+" "+str(idx)+" "+str(mask_vector)
          gt_bbox.append(mask_vector)

      bbox_per_batch.append(gt_bbox)
      boundary_adhesions_per_batch.append(boundary_adhesion_pre)

      aidx_per_image, delta_per_image = [], []
      aidx_set = set()
      for i in range(len(gt_bbox)):
        overlaps = batch_iou(mc.ANCHOR_BOX, gt_bbox[i])
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
          dist = np.sum(np.square(gt_bbox[i] - mc.ANCHOR_BOX), axis=1)
          for dist_idx in np.argsort(dist):
            if dist_idx not in aidx_set:
              aidx_set.add(dist_idx)
              aidx = dist_idx
              break
        if mc.EIGHT_POINT_REGRESSION:
          box_cx, box_cy, box_w, box_h, of1, of2, of3, of4 = gt_bbox[i]
          delta = [0]*8
        else:
          box_cx, box_cy, box_w, box_h = gt_bbox[i]
          delta = [0]*4

        if mc.ASYMMETRIC_ENCODING:
          # Use spatial domain anchors
          xmin_t, ymin_t, xmax_t, ymax_t = bbox_transform([box_cx, box_cy, box_w, box_h])
          xmin_a, ymin_a, xmax_a, ymax_a = bbox_transform(mc.ANCHOR_BOX[aidx])
          delta[0] = (xmin_t - xmin_a)/mc.ANCHOR_BOX[aidx][2]
          delta[1] = (ymin_t - ymin_a)/mc.ANCHOR_BOX[aidx][3]
          delta[2] = (xmax_t - xmax_a)/mc.ANCHOR_BOX[aidx][2]
          delta[3] = (ymax_t - ymax_a)/mc.ANCHOR_BOX[aidx][3]
        else:
          delta[0] = (box_cx - mc.ANCHOR_BOX[aidx][0])/mc.ANCHOR_BOX[aidx][2]
          delta[1] = (box_cy - mc.ANCHOR_BOX[aidx][1])/mc.ANCHOR_BOX[aidx][3]
          delta[2] = np.log(box_w/mc.ANCHOR_BOX[aidx][2]) # if box_w or box_h = 0, the box is not included
          delta[3] = np.log(box_h/mc.ANCHOR_BOX[aidx][3])

        if mc.EIGHT_POINT_REGRESSION:
          assert not mc.ASYMMETRIC_ENCODING, "ASYMMETRIC_ENCODING not supported with EIGHT_POINT_REGRESSION"
          EPSILON = 1e-8
          anchor_diagonal = (mc.ANCHOR_BOX[aidx][2]**2+mc.ANCHOR_BOX[aidx][3]**2)**(0.5)
          delta[4] = np.log((of1 + EPSILON)/anchor_diagonal)
          delta[5] = np.log((of2 + EPSILON)/anchor_diagonal)
          delta[6] = np.log((of3 + EPSILON)/anchor_diagonal)
          delta[7] = np.log((of4 + EPSILON)/anchor_diagonal)

        aidx_per_image.append(aidx)
        delta_per_image.append(delta)

      delta_per_batch.append(delta_per_image)
      aidx_per_batch.append(aidx_per_image)

    if mc.DEBUG_MODE:
      print ('max iou: {}'.format(max_iou))
      print ('min iou: {}'.format(min_iou))
      print ('avg iou: {}'.format(avg_ious/num_objects))
      print ('number of objects: {}'.format(num_objects))
      print ('number of objects with 0 iou: {}'.format(num_zero_iou_obj))

    return image_per_batch, label_per_batch, delta_per_batch, \
        aidx_per_batch, bbox_per_batch, boundary_adhesions_per_batch



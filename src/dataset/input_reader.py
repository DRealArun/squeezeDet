# Author: Arun Prabhu (arun.rajendra.prabhu@iais.fraunhofer.de) 10/11/2019

"""The data base wrapper class around imdb to override the read_batch function"""

import os
import random
import shutil
import math
import pywt

from dataset.imdb import imdb
from PIL import Image, ImageFont, ImageDraw
import cv2
import numpy as np
from utils.util import iou, batch_iou, get_distance_measure
from utils.polysimplify import VWSimplifier
from skimage import measure

def get_wavelet_transform(contour_points, transform):
  # Do error check for type of transform
  db1 = pywt.Wavelet(transform) # TODO : dont hard-code HAAR
  x = contour_points[:,0]
  y = contour_points[:,1]
  # https://en.wikipedia.org/wiki/Discrete_wavelet_transform
  # cD : Detail coefficient output of high pass filtering
  # cA : Approximation coefficient output of low pass filtering
  _x_coeffs = pywt.wavedec(x, db1, mode='periodic', level=1) # Returns (cA, cD) hence (_x_coeffs[0], _x_coeffs[1]) = (cA, cD) 
  _y_coeffs = pywt.wavedec(y, db1, mode='periodic', level=1) # Returns (cA, cD)  
#     print("Number of contour points:",np.shape(contour_points)[0])
#     print("Length of the coeffs_x and coeffs_y:",len(_x_coeffs))
  return _x_coeffs, _y_coeffs

def get_keypoint_indices(contour_points, plot='True', threshold_val=None, transform='haar'):
  keypoint_indices = []
  coeffs_x, coeffs_y = get_wavelet_transform(contour_points, transform)
  for i in range(len(coeffs_x)):
      coeff_x = np.square(coeffs_x[i])
      coeff_y = np.square(coeffs_y[i])
      variation = np.sqrt(coeff_x + coeff_y)
      if threshold_val == None:
          threshold = np.mean(variation)
          print("Theshold", threshold)
      else:
          threshold = threshold_val
      if i > 0: # Select only values from the high frequency coefficients
          keypoint_indices.append(np.where(variation > threshold))
      if plot:
          f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,7))
          if i == 0:
              f.suptitle("Low frequency co-efficients")
          if i == 1:
              f.suptitle("High frequency co-efficients")
          ax1.plot(range(len(coeff_x)), variation, label='total_variation')
          ax1.plot(range(len(coeff_x)), [threshold]*len(coeff_x), label='mean_variation')
          ax1.legend()
          ax1.grid()
          ax2.plot(range(len(coeff_x)), coeffs_x[i], label='x')
          ax2.plot(range(len(coeff_x)), coeffs_y[i], label='y')
          ax2.legend()
          ax2.grid()
          plt.show()
  return keypoint_indices, coeffs_x, coeffs_y

def check_if_significant_with_slope_1(sel, val, stepx=10, stepy=10, slope_thresh=10**(-3)): # filter based on distance from last keypoint
  angle = 0
  if len(sel) != 0:
      diff = np.square(np.subtract(sel[-1], val))
      dist = np.sqrt(np.sum(diff))
      angle = np.arctan(diff[1]/(diff[0]+1*10-5)) * 180 / np.pi
      flag = 1
      if angle < slope_thresh or (90-angle) < slope_thresh:
          if dist < (stepx**2+stepy**2)**(0.5):
              flag = 0
  else:
      flag = 1
  return flag


def fill_missing_values(selected, whole_set, max_num_elements):
  # The selected list contains number of elements less than max_num_elements
  # Find sum of the distances between each element in whole_set (which are not in selected list) and all the
  # elements in the selected set.
  # Arrange the values in whole_set in descending value and select the first (max_num_elements-len(selected)) 
  # elements
  selected = [(h[0],h[1]) for h in selected]
  whole_set = [(h[0],h[1]) for h in whole_set]
  newly_selected = []
  distances = []
  num_missing_values = max_num_elements-len(selected)
  set_whole = set(whole_set)
  set_selected = set(selected)
  set_not_selected = set_whole-set_selected
  if len(set_not_selected) == num_missing_values:
      final_selected = list(set_whole) # would not come here because of the check before this function is called
  elif len(set_not_selected) > num_missing_values:
      for pt in set_not_selected:
          distances.append(np.sum(np.sqrt(np.sum(np.square(np.subtract(selected, pt)), axis=1))))
      indices = np.argsort(distances)[::-1]
      set_not_selected = list(set_not_selected)
      newly_selected = [set_not_selected[idx] for idx in indices[0:num_missing_values]]
      final_selected = [[]] * max_num_elements
      final_selected[0:len(selected)] = selected
      final_selected[len(selected):] = newly_selected
      order = [whole_set.index(ele) for ele in final_selected]
      final_order_indices = np.argsort(order)
      final_selected = np.array(final_selected)[final_order_indices].tolist()
  else:
      # maybe interpolate for later
      set_whole_copy = list(set_whole)
      final_selected = list(set_whole)
      for count in range(max_num_elements-len(set_whole_copy)):
          index_to_insert = random.choice(range(len(set_whole_copy)))
          final_selected.insert(index_to_insert, set_whole_copy[index_to_insert])
  return final_selected

def subsample_important_values(selected, max_num_elements):
  simplifier = VWSimplifier(selected)
  final_list = simplifier.from_number(max_num_elements)
  return final_list

def draw_mask(contour_poly, dim):
  shuffled = np.zeros_like(contour_poly)
  shuffled[:,0] = contour_poly[:,1]
  shuffled[:,1] = contour_poly[:,0]
  mask_from_contour = np.zeros(dim)
  cv2.drawContours(mask_from_contour, np.int32([shuffled]), -1, (255),thickness=-1)
  return mask_from_contour

def get_cropped_mask_and_contour(mask, poly, margin=2):
  imH, imW = np.shape(mask)
  xmin = int(round(max(min(poly[:,0])-2, 0)))
  xmax = int(round(min(max(poly[:,0])+2, imW)))
  ymin = int(round(max(min(poly[:,1])-2, 0)))
  ymax = int(round(min(max(poly[:,1])+2, imH)))
  mask_cropped = mask[ymin:ymax,xmin:xmax]
  contours_original = measure.find_contours(mask_cropped, 0.8)
  contours = []
  min_len = 0
  if len(contours_original) == 1:
    contours = np.squeeze(contours_original)
  if len(contours_original) != 1:
    # HAVE TO CHANGE THIS (TODO)
    for con in contours_original:
      if len(con) > min_len:
        min_len = len(con)
        contours = np.squeeze(con)
  return mask_cropped, contours, xmin, ymin
    
def _get_key_point_encoding(contours_to_analyse, mask_to_analyse, xmin, ymin, max_num_key_points):
  indices, coeffs_x, coeffs_y = get_keypoint_indices(contours_to_analyse, threshold_val=0, plot=False)
  indices = np.squeeze(indices)
  selection = []
  stepx_val = np.shape(mask_to_analyse)[0]/(32)
  stepy_val = np.shape(mask_to_analyse)[1]/(32)
  for ida in indices:
      if check_if_significant_with_slope_1(selection, contours_to_analyse[ida*2], stepx=stepx_val, stepy=stepy_val, slope_thresh=30):
          selection.append(contours_to_analyse[ida*2])
  first_ = len(selection)
  second_ = 0
  third_ = 0
  fourth_ = 0
  if len(selection) < max_num_key_points:
      selection = fill_missing_values(selection, contours_to_analyse, max_num_key_points)
      second_ = len(selection)
  elif len(selection) > max_num_key_points:
      selection = subsample_important_values(selection, max_num_key_points)
      third_ = len(selection)
      if len(selection) < max_num_key_points: # can happen
          selection = fill_missing_values(selection, contours_to_analyse, max_num_key_points)
          fourth_ = len(selection)
  selection = np.asarray(selection)
  # Since selection is obtained from PIL contour finding function which gives (rows, col) array.
  index_x_min = np.argsort(selection[:,1])[0] # selection is (row, col) array
  origin = selection[index_x_min]
  center_x = xmin + origin[1]
  center_y = ymin + origin[0]
  x_diffs = np.subtract(selection[:,1],origin[1])
  y_diffs = np.subtract(selection[:,0],origin[0])
  sinThetas = y_diffs/(np.sqrt(np.add(np.square(x_diffs),np.square(y_diffs)))+1*10**-5)
  sinThetas_reduced = np.hstack((sinThetas[index_x_min+1:], sinThetas[0:index_x_min]))
  x_diffs_reduced = np.hstack((x_diffs[index_x_min+1:], x_diffs[0:index_x_min]))
  if len(sinThetas_reduced) != 19 or len(x_diffs_reduced) != 19:
    print("Value error while extracting", index_x_min, len(selection), first_, second_, third_, fourth_, len(contours_to_analyse))
  return center_x, center_y, sinThetas_reduced, x_diffs_reduced

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
  m1 = (pts[2][1]-pts[0][1])/(pts[2][0]-pts[0][0])
  m2 = (pts[3][1]-pts[1][1])/(pts[3][0]-pts[1][0])
  ms = [-1/m1, -1/m2, -1/m2, -1/m1, -1/m1, -1/m2, -1/m2, -1/m1]
  intersecting_pts = []
  for eq1, pt, vert_hor, m in zip(eq1s, points, vert_or_hors, ms):
      op_pt = _get_intersecting_point(vert_hor, eq1, pt, m)
      intersecting_pts.append(op_pt)
  return intersecting_pts

def get_mask(polygon, imh, imw, divider=1):
  polygon = [polygon]
  mask = np.zeros((imh,imw))
  mask = cv2.fillPoly(mask, np.int32(polygon), color=255)
  mask_reshaped = cv2.resize(mask, (imw//divider, imh//divider), interpolation = cv2.INTER_AREA)
  ret, mask_reshaped = cv2.threshold(mask_reshaped, 127, 255, 0)
  return mask_reshaped

class input_reader(imdb):
  """Base image database handler for instance based annotations."""

  def __init__(self, name, mc):
    imdb.__init__(self, name, mc)


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
    rr, cc = outline[:,1], outline[:,0]
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

  def read_batch(self, shuffle=True):
    """Read a batch of image and instance annotations.
    Args:
      shuffle: whether or not to shuffle the dataset
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
      if mc.EIGHT_POINT_REGRESSION or mc.MULTI_POINT_REGRESSION:
        polygons = [b[2] for b in self._poly[idx][:]]
      is_drift_performed = False
      is_flip_performed = False

      if mc.DATA_AUGMENTATION:
        assert mc.DRIFT_X >= 0 and mc.DRIFT_Y > 0, \
            'mc.DRIFT_X and mc.DRIFT_Y must be >= 0'

        if mc.DRIFT_X > 0 or mc.DRIFT_Y > 0:
          # Ensures that gt boundibg box is not cut out of the image
          max_drift_x = min(gt_bbox_pre[:, 0] - gt_bbox_pre[:, 2]/2.0+1)
          max_drift_y = min(gt_bbox_pre[:, 1] - gt_bbox_pre[:, 3]/2.0+1)
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
          im = distorted_im

        # Flip image with 50% probability
        if np.random.randint(2) > 0.5:
          im = im[:, ::-1, :]
          is_flip_performed = True
          gt_bbox_pre[:, 0] = orig_w - 1 - gt_bbox_pre[:, 0]

      # scale image
      im = cv2.resize(im, (mc.IMAGE_WIDTH, mc.IMAGE_HEIGHT))
      image_per_batch.append(im)

      # scale annotation
      x_scale = mc.IMAGE_WIDTH/orig_w
      y_scale = mc.IMAGE_HEIGHT/orig_h
      gt_bbox_pre[:, 0::2] = gt_bbox_pre[:, 0::2]*x_scale
      gt_bbox_pre[:, 1::2] = gt_bbox_pre[:, 1::2]*y_scale
      if mc.EIGHT_POINT_REGRESSION or mc.MULTI_POINT_REGRESSION:
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
            print("Error width and height", width, height, gt_bbox_pre[k][2], gt_bbox_pre[k][3], center_x, center_y, gt_bbox_pre[k][0], gt_bbox_pre[k][1], idx)
            # del label_per_batch[img_ct][k]
            # continue #ONLY FOR COMPLETE SET OF CLASSES
          assert not (of1 < 0 or of2 < 0 or of3 < 0 or of4 < 0), "Error Occured "+ str(of1) +" "+ str(of2)+" "+ str(of3)+" "+ str(of4)
          points = decode_parameterization(mask_vector)
          points = np.round(points)
          points = np.array(points, 'int32')
          assert not (points[0][1] - points[1][1] > 1 or points[2][0] - points[3][0] > 1 or points[5][1] - points[4][1] > 1 or points[7][0] - points[6][0] > 1), "\n\n Error in extraction:"+str(points)+" "+str(idx)+" "+str(mask_vector)
          gt_bbox.append(mask_vector)

      if mc.MULTI_POINT_REGRESSION:
        num_mask_params_local = 20
        gt_bbox = []
        actual_bin_masks = []
        for k in range(len(polygons)):
          polygon = polygons[k]
          msk_from_poly = get_mask(np.around(polygon,2), mc.IMAGE_HEIGHT, mc.IMAGE_WIDTH, 1)
          mask_cropped, contours_squeezed, x_min, y_min = get_cropped_mask_and_contour(msk_from_poly, polygon)
          assert len(contours_squeezed) != 0, "Error in contours "+ str(len(contours_squeezed))
          centerx, centery, sin, x_differences = _get_key_point_encoding(contours_squeezed, mask_cropped, x_min, y_min, num_mask_params_local)
          individual_box = [centerx, centery]
          individual_box.extend(sin.tolist())
          individual_box.extend(x_differences.tolist())
          gt_bbox.append(individual_box)

      bbox_per_batch.append(gt_bbox)

      aidx_per_image, delta_per_image = [], []
      aidx_set = set()
      for i in range(len(gt_bbox)):
        if not mc.MULTI_POINT_REGRESSION:
          encompassing_box = gt_bbox[i][0:4]
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
          delta[0] = (box_cx - mc.ANCHOR_BOX[aidx][0])/mc.ANCHOR_BOX[aidx][2]
          delta[1] = (box_cy - mc.ANCHOR_BOX[aidx][1])/mc.ANCHOR_BOX[aidx][3]
          delta[2] = np.log(box_w/mc.ANCHOR_BOX[aidx][2])
          delta[3] = np.log(box_h/mc.ANCHOR_BOX[aidx][3])

          if mc.EIGHT_POINT_REGRESSION:
            EPSILON = 1e-8
            anchor_diagonal = (mc.ANCHOR_BOX[aidx][2]**2+mc.ANCHOR_BOX[aidx][3]**2)**(0.5)
            delta[4] = np.log((of1 + EPSILON)/anchor_diagonal)
            delta[5] = np.log((of2 + EPSILON)/anchor_diagonal)
            delta[6] = np.log((of3 + EPSILON)/anchor_diagonal)
            delta[7] = np.log((of4 + EPSILON)/anchor_diagonal)
        else:
          overlaps = get_distance_measure(mc.ANCHOR_BOX, gt_bbox[i])
          # aidx = len(mc.ANCHOR_BOX)
          for ov_idx in np.argsort(overlaps):
            if ov_idx not in aidx_set:
              aidx_set.add(ov_idx)
              aidx = ov_idx
              break
          box_cx = gt_bbox[i][0]
          box_cy = gt_bbox[i][1]
          box_sin = gt_bbox[i][2:2+(num_mask_params_local)-1]
          box_x_differences = gt_bbox[i][2+(num_mask_params_local)-1:]
          delta = [0]*2*num_mask_params_local
          delta[0] = (box_cx - mc.ANCHOR_BOX[aidx][0])/16
          delta[1] = (box_cy - mc.ANCHOR_BOX[aidx][1])/16
          delta[2:2+(num_mask_params_local)-1] = box_sin
          delta[2+(num_mask_params_local)-1:] = np.log10(np.add(box_x_differences,1))
          if len(delta) != 40:
            print("Lenght of last value", len(delta))
            print(delta[0], delta[1], delta[2:2+(num_mask_params_local)-1], delta[2+(num_mask_params_local)-1:], num_mask_params_local)
            print(len(box_sin), len(np.log10(np.add(box_x_differences,1))))

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
        aidx_per_batch, bbox_per_batch



# Author: Arun Prabhu (arun.rajendra.prabhu@iais.fraunhofer.de) 10/11/2019

"""The data base wrapper class around imdb to override the read_batch function"""

import os
import random
import shutil
import math

from dataset.imdb import imdb
from PIL import Image, ImageFont, ImageDraw
import cv2
import numpy as np
from utils.util import iou, batch_iou
import pywt
from dataset.polysimplify_1 import VWSimplifier
from skimage import measure
import copy

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

  def get_importance_values(self, values, debug=False):
    imp_list = []
    for i in range(len(values)):
        prev_val = values[i-1]
        next_val = values[(i+1)%len(values)]
        curr_val = values[i]
        imp_val = (curr_val-prev_val) + (curr_val-next_val) # If it stands out, it is important
        if debug:
            print("Prev:", prev_val, "Curr:", curr_val, "Next:", next_val, "Imp Metric:", imp_val)
        imp_list.append(imp_val)
    return imp_list

  def draw_mask(self, contour_poly, dim):
    swapped = np.zeros_like(contour_poly)
    swapped[:,0] = contour_poly[:,1]
    swapped[:,1] = contour_poly[:,0]
    mask_from_contour = np.zeros(dim)
    cv2.drawContours(mask_from_contour, np.int32([swapped]), -1, (255),thickness=-1)
    return mask_from_contour

  def subsample_important_values(self, selected, max_num_elements):
    simplifier = VWSimplifier(selected)
    final_list, indices = simplifier.from_number(max_num_elements)
    return indices

  def get_wavelet_transform(self, contour_points):
    db1 = pywt.Wavelet('haar') # TODO : dont hard-code HAAR
    x = contour_points[:,0]
    y = contour_points[:,1]
    # https://en.wikipedia.org/wiki/Discrete_wavelet_transform
    # cD : Detail coefficient output of high pass filtering
    # cA : Approximation coefficient output of low pass filtering
    _x_coeffs = pywt.wavedec(x, db1, mode='periodic', level=1) # Returns (cA, cD) hence (_x_coeffs[0], _x_coeffs[1]) = (cA, cD) 
    _y_coeffs = pywt.wavedec(y, db1, mode='periodic', level=1) # Returns (cA, cD)  
    return _x_coeffs, _y_coeffs

  def check_if_significant_with_slope_1(self, sel, val, stepx=10, stepy=10, slope_thresh=10**(-3)): # filter based on distance from last keypoint
    angle = 0
    if len(sel) != 0:
      diff = np.square(np.subtract(sel[-1], val))
      dist = np.sqrt(np.sum(diff))
      if diff[0] == 0:
        angle = 90
      else:
        angle = np.arctan(diff[1]/diff[0]) * 180 / np.pi
      # print("diff", diff, angle)
      flag = 1
      if angle < slope_thresh or (90-angle) < slope_thresh:
        if dist < (stepx**2+stepy**2)**(0.5):
          flag = 0
    else:
      flag = 1
    return flag

  def wavelet_based_key_point_extractor_3(self, values, max_num, dim=(512, 1024)):
    coeffs_x, coeffs_y = self.get_wavelet_transform(values)
    if len(coeffs_x[0]) == max_num: #less than max_num is unlikely
      Ax_ = coeffs_x[0] # Gather low freq co-efficents along x
      Ay_ = coeffs_y[0] # Gather low freq co-efficents along y
      # print("Length of list", len(Ax_))
      indices = list(range(len(coeffs_x[0])))
      indices = np.squeeze(indices)
      return Ax_, Ay_, values[indices*2]
    else:
      indices = list(range(len(coeffs_x[0])))
      indices = np.squeeze(indices)
      # print("After DWT", len(indices))
      selection = []
      id_list = []
      stepx_val = dim[1]/(16)
      stepy_val = dim[0]/(16)
      for ida in indices:
        if self.check_if_significant_with_slope_1(selection, values[ida*2], stepx=stepx_val, stepy=stepy_val, slope_thresh=30):
          selection.append(values[ida*2])
          id_list.append(ida)
      # print("After First Stage filtering", len(selection))
      sec_stage_indices = self.subsample_important_values(selection, max_num)[0]
      reduced_id_list = []
      for sec_id in sec_stage_indices:
        ida = id_list[sec_id]
        reduced_id_list.append(ida)
      id_list = reduced_id_list
      # print("After Second Stage filtering", len(id_list))
      if len(reduced_id_list) < max_num:
        # Select values from the list obtained after first stage of filtering
        distances = []
        whole_list = set(id_list)                
        current_selection = set(reduced_id_list)
        not_selected = whole_list - current_selection
        num_missing_vals = max_num - len(current_selection)
        if len(not_selected) < num_missing_vals:
          # Select values from the list obtained after extracting the indices
          whole_list = set(indices)
          current_selection = set(reduced_id_list)
          not_selected = whole_list - current_selection
          num_missing_vals = max_num - len(current_selection)
        if len(not_selected) > num_missing_vals:
          not_selected_points = values[list(not_selected)]
          selected_points = values[list(current_selection)]
          for pt in not_selected_points:
            distances.append(np.sum(np.sqrt(np.sum(np.square(np.subtract(selected_points, pt)), axis=1))))
          sorted_indices = np.argsort(distances)[::-1]
          not_selected = list(not_selected)
          newly_selected = [not_selected[idx] for idx in sorted_indices[0:num_missing_vals]]
          final_selected = [[]] * max_num
          final_selected[0:len(current_selection)] = current_selection
          final_selected[len(selected_points):] = newly_selected
          whole_list = list(whole_list)
          order = [whole_list.index(ele) for ele in final_selected]
          final_order_indices = np.argsort(order)
          final_selected = np.array(final_selected)[final_order_indices].tolist()
        else:
          whole_list = list(whole_list)
          final_selected = copy.deepcopy(whole_list)
          for count in range(max_num-len(whole_list)):
            index_to_insert = random.choice(range(len(whole_list)))
            final_selected.insert(index_to_insert, whole_list[index_to_insert])
          order = [whole_list.index(ele) for ele in final_selected]
          final_order_indices = np.argsort(order)
          final_selected = np.array(final_selected)[final_order_indices].tolist()    
        id_list = final_selected
        # print("Length of list", len(id_list))
      Ax_ = coeffs_x[0][id_list] # Gather low freq co-efficents along x
      Ay_ = coeffs_y[0][id_list] # Gather low freq co-efficents along y
      # print("Length of list", len(Ax_))
      return Ax_, Ay_, values[np.asarray(id_list)*2]

  def get_mask(self, polygon, imh, imw, divider=1):
    polygon = np.round(polygon) # Ensure rounding
    polygon = np.int32(polygon)
    polygon = [polygon]
    mask = np.zeros((imh,imw))
    mask = cv2.fillPoly(mask, polygon, color=255)
    mask_reshaped = cv2.resize(mask, (imw//divider, imh//divider), interpolation = cv2.INTER_AREA)
    ret, mask_reshaped = cv2.threshold(mask_reshaped, 127, 255, 0)
#     print(np.unique(mask_reshaped))
    return mask_reshaped

  def get_key_points(self, polymask, max_num, dim, classes):
    h, w = dim
    outline = polymask
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

    # Create mask from polygon
    mask_image = self.get_mask(polymask, dim[0], dim[1], divider=1)
    xmin = int(round(max(min(polymask[:,0])-2, 0)))
    xmax = int(round(min(max(polymask[:,0])+2, dim[1])))
    ymin = int(round(max(min(polymask[:,1])-2, 0)))
    ymax = int(round(min(max(polymask[:,1])+2, dim[0])))
    mask_cropped = mask_image[ymin:ymax,xmin:xmax]
    # Extract the contour
    contours = measure.find_contours(mask_cropped, 0.8)
    if len(contours) != 1:
      # print("Length of contours", len(contours))
      max_area = 0
      max_id = 0
      for i, contour in enumerate(contours):
        wd = max(contour[:,0])-min(contour[:,0])
        hd = max(contour[:,1])-min(contour[:,1])
        area = wd*hd
        if area > max_area:
          max_area = area
          max_id = i
      swapped_version = np.squeeze(contours[max_id])
    else:
      swapped_version = np.squeeze(contours)

    
    segment_list, segment_lengths = self.segment_contour(swapped_version, 10)
    values = self.get_base_line_contour(segment_list, segment_lengths)
    keypoints = np.zeros_like(values)
    keypoints[:,0] = values[:,1]
    keypoints[:,1] = values[:,0]
    r, angles = self.get_polar_coords(keypoints, xmin, ymin)
    labels = []
    for angle in angles:
        for idx, class_val in enumerate(classes):
            if angle >= int(class_val):
                label = idx
        labels.append(label)
    
    mask_vector = [0]*24
    mask_vector[0] = center_x
    mask_vector[1] = center_y
    mask_vector[2] = width
    mask_vector[3] = height
    # for l, v in enumerate(Ax_):
    #   mask_vector[4+l] = v
    # for l, v in enumerate(Ay_):
    #   mask_vector[14+l] = v
    for l in range(len(r)):
      # print(l, len(r), len(labels))
      mask_vector[4+l] = r[l]
      mask_vector[14+l] = labels[l]
    return mask_vector

  def wavelet_based_key_point_extractor_2(self, polygon, h, w):
    max_num=10
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

    db1 = pywt.Wavelet('haar') # TODO : dont hard-code HAAR
    x = outline[:,0] - xmin
    y = outline[:,1] - ymin
    Ax_, Dx_ = pywt.wavedec(x, db1, mode='periodic', level=1)
    Ay_, Dy_ = pywt.wavedec(y, db1, mode='periodic', level=1)
    metric_values_x = self.get_importance_values(Dx_, False)
    metric_values_y = self.get_importance_values(Dy_, False)
    consolidated_metrics = [max(a,b) for a,b in zip(metric_values_x, metric_values_y)]
    sorted_metric_indices = np.argsort(consolidated_metrics)
    # Returns the shortlisted co-effs for x and y
    mask_vector = [0]*24
    mask_vector[0] = center_x
    mask_vector[1] = center_y

    for l, v in enumerate(Ax_[sorted_metric_indices[0:max_num]]):
      mask_vector[2+l] = v
    for l, v in enumerate(Ay_[sorted_metric_indices[0:max_num]]):
      mask_vector[12+l] = v

    # if width or height > 100:
    #   db1_1 = pywt.Wavelet('haar')
    #   v_ = [Ax_[sorted_metric_indices[0:max_num]], np.zeros_like(Ax_[sorted_metric_indices[0:max_num]])]
    #   w_ = [Ay_[sorted_metric_indices[0:max_num]], np.zeros_like(Ay_[sorted_metric_indices[0:max_num]])]
    #   reconstructed_x_rel = pywt.waverec(v_, db1_1) # Reconstruct the points x co-ord
    #   reconstructed_y_rel = pywt.waverec(w_, db1_1) # Reconstruct the points y co-ord
    #   center_x_1 = (max(reconstructed_x_rel) - min(reconstructed_x_rel))/2
    #   center_y_1 = (max(reconstructed_y_rel) - min(reconstructed_y_rel))/2
    #   x_vals = np.reshape(reconstructed_x_rel - center_x_1 + center_x, (-1,1))
    #   y_vals = np.reshape(reconstructed_y_rel - center_y_1 + center_y, (-1,1))
    #   contour = np.hstack((y_vals, x_vals)) # y first then x
    #   contour = np.round(contour) # Ensure rounding
    #   contour = np.array(contour, 'int32')
    #   polygon1 = [contour]
    #   im = np.zeros((h,w))
    #   mask = cv2.fillPoly(im, polygon1, color=255)
    #   cv2.imwrite("inspection.jpg", im)
    #   print("Done Saving !")

    return mask_vector


  def get_polar_coords(self, keypoints_values, x_orig, y_orig):
    # xmin, ymin is the origin
    r = np.sqrt(np.sum(np.square(keypoints_values), axis=1))
    angles = []
    for key_pt in keypoints_values:
      if key_pt[0] > 0:
        angle = np.arctan(key_pt[1]/key_pt[0]) * 180.0 / np.pi
      else:
        angle = 90.0
      angles.append(angle)
    return r, np.asarray(angles)

  def get_cart_coords(self, r, angles, x_orig, y_orig):
    # xmin, ymin is the origin
    x_vals = np.reshape(np.multiply(r, np.cos(angles*np.pi/180.0)), (-1,1))
    y_vals = np.reshape(np.multiply(r, np.sin(angles*np.pi/180.0)), (-1,1))
    contour = np.hstack((x_vals, y_vals)) # y first then x
    contour = np.round(contour) # Ensure rounding
    contour = np.array(contour, 'int32')
    return contour

  def close_contour_if_necessary(self, points):
    if points[-1][0] != points[0][0] or points[-1][1] != points[0][1]:
      # print("Contour is not closed, so closing it !")
      diff = points[0] - points[-1]
      if diff[0] == 0:
        inc = 1
        if diff[1] < 0:
          inc = -1
        fillers = [(points[0][0], y) for y in np.arange(points[-1][1], points[0][1], inc)] #Since step is 1 the last point does not reach the destination but gets close
        fillers.append((points[0][0], points[0][1])) # So append the destination
      else:
        slope = diff[1]/diff[0]
        c = points[0][1] - (slope*points[0][0])
        inc = 1
        if diff[0] < 0:
          inc = -1
        fillers = [(x, (slope*x)+c) for x in np.arange(points[-1][0], points[0][0], inc)] #Since step is 1 the last point does not reach the destination but gets close
        fillers.append((points[0][0], points[0][1])) # So append the destination
      num_points = len(points) + len(fillers)
      # print(num_points, len(points), len(fillers), points[-1], points[0])
      filled_contour = np.zeros((num_points, 2))
      filled_contour[0:len(points),:] = points
      filled_contour[len(points):,:] = np.asarray(fillers)
    else:
      filled_contour = points
    return filled_contour
      
  def segment_contour(self, points, num_seg):
    points = self.close_contour_if_necessary(points)
    contours_squeezed_shifted = np.roll(points, -1, axis=0)
    perimeter = np.sum(np.sqrt(np.sum(np.square(points-contours_squeezed_shifted), axis=1)))
    seg_len = perimeter/num_seg
    # print("Perimeter Local:", perimeter, seg_len, num_seg)
    dist_counter = 0
    seg_list = []
    dist_list = []
    seg = []
    dist = 0
    last_dist = 0
    lim = seg_len
    for pt1, pt2 in zip(points, contours_squeezed_shifted):
      dist += np.sqrt(np.sum(np.square(pt1-pt2)))
      seg.append(pt1)
      if dist >= lim:
        seg_list.append(np.asarray(seg))
        dist_list.append(dist-last_dist)
        seg = []
        lim += seg_len
        last_dist = dist
    if len(seg) != 0 and dist-last_dist != 0:
      # print(len(seg), dist)
      seg_list.append(np.asarray(seg))
      dist_list.append(dist-last_dist)
    # print("Num of segments:", len(seg_list), dist_list, np.sum(dist_list))
    return seg_list, dist_list
              
  def get_base_line_contour(self, seg_list, dist_list):
    keypoint_list = []
    for seg, dist in zip(seg_list, dist_list):
      inc_dist = 0
      seg_shifted = np.roll(seg, -1, axis=0)
      for pt1, pt2 in zip(seg, seg_shifted):
        inc_dist += np.sqrt(np.sum(np.square(pt1-pt2)))
        if inc_dist >= dist/2:
          keypoint_list.append(pt1)
          break
    # print("Number of keypoints:", len(keypoint_list))
    return np.asarray(keypoint_list)

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
    angles_per_batch = []
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
      if mc.EIGHT_POINT_REGRESSION_1:
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
      if mc.EIGHT_POINT_REGRESSION_1:
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
      gt_bbox = gt_bbox_pre # Use shifted bounding box if EIGHT_POINT_REGRESSION_1 = False
      # Transform the bounding box to offset mode.
      # We extract the bounding box from the flipped and drifted masks to ensure
      # consistency.
      if mc.EIGHT_POINT_REGRESSION_1:

        gt_bbox = []
        gt_angle_bbox = []
        for k in range(len(polygons)):
          polygon = polygons[k]
          # mask_vector = self.wavelet_based_key_point_extractor_2(polygon, mc.IMAGE_HEIGHT, mc.IMAGE_WIDTH)
          mask_vector = self.get_key_points(polygon, 10, (mc.IMAGE_HEIGHT, mc.IMAGE_WIDTH), self._angle_classes)
          # center_x, center_y, width, height, of1, of2, of3, of4 = mask_vector
          # if width == 0 or height == 0:
          #   print("Error in width or height", width, height, gt_bbox_pre[k][2], gt_bbox_pre[k][3], center_x, center_y, gt_bbox_pre[k][0], gt_bbox_pre[k][1], idx)
          #   del label_per_batch[img_ct][k]
          #   continue
          # assert not (of1 <= 0 or of2 <= 0 or of3 <= 0 or of4 <= 0), "Error Occured "+ str(of1) +" "+ str(of2)+" "+ str(of3)+" "+ str(of4)
          # points = decode_parameterization(mask_vector)
          # points = np.round(points)
          # points = np.array(points, 'int32')
          # assert not ((points[0][1] - points[1][1]) > 1 or (points[2][0] - points[3][0]) > 1 or (points[5][1] - points[4][1]) > 1 or (points[7][0] - points[6][0]) > 1), \
          #   "\n\n Error in extraction:"+str(points)+" "+str(idx)+" "+str(mask_vector)
          gt_bbox.append(mask_vector[0:14])
          gt_angle_bbox.append(mask_vector[14:])

      bbox_per_batch.append(gt_bbox)
      angles_per_batch.append(gt_angle_bbox)

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
          dist = np.sum(np.square(gt_bbox[i][0:2] - mc.ANCHOR_BOX), axis=1)
          for dist_idx in np.argsort(dist):
            if dist_idx not in aidx_set:
              aidx_set.add(dist_idx)
              aidx = dist_idx
              break
        if mc.EIGHT_POINT_REGRESSION_1:
          # box_cx, box_cy, box_w, box_h, of1, of2, of3, of4 = gt_bbox[i]
          box_cx = gt_bbox[i][0]
          box_cy = gt_bbox[i][1]
          box_w = gt_bbox[i][2]
          box_h = gt_bbox[i][3]
          delta = [0]*14
        else:
          box_cx, box_cy, box_w, box_h = gt_bbox[i]
          delta = [0]*4

        delta[0] = (box_cx - mc.ANCHOR_BOX[aidx][0])/mc.ANCHOR_BOX[aidx][2]
        delta[1] = (box_cy - mc.ANCHOR_BOX[aidx][1])/mc.ANCHOR_BOX[aidx][3]
        delta[2] = np.log(box_w/mc.ANCHOR_BOX[aidx][2]) # if box_w or box_h = 0, the box is not included
        delta[3] = np.log(box_h/mc.ANCHOR_BOX[aidx][3])

        if mc.EIGHT_POINT_REGRESSION_1:
          EPSILON = 1e-8
          anchor_diagonal = (mc.ANCHOR_BOX[aidx][2]**2+mc.ANCHOR_BOX[aidx][3]**2)**(0.5)
          for l in range(10):
            # delta[4+l] = math.log(gt_bbox[i][4+l] + EPSILON)
            # delta[14+l] = math.log(gt_bbox[i][14+l] + EPSILON)
            delta[4+l] = math.log((gt_bbox[i][4+l] + EPSILON)/anchor_diagonal)
            # delta[14+l] = math.log((gt_bbox[i][14+l] + EPSILON)/mc.ANCHOR_BOX[aidx][3])
          # delta[6] = np.log((of3 + EPSILON)/anchor_diagonal)
          # delta[7] = np.log((of4 + EPSILON)/anchor_diagonal)

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
        aidx_per_batch, bbox_per_batch, angles_per_batch
# Author: Bichen Wu (bichen@berkeley.edu) 08/25/2016

"""Image data base class for toy_car"""

import cv2
import os 
import numpy as np
import subprocess
from PIL import Image

from dataset.imdb import imdb
from utils.util import bbox_transform_inv, batch_iou
import imageio as sp
import json

from collections import namedtuple
Label = namedtuple( 'Label' , [

    'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                    # We use them to uniquely name a class

    'id'          , # An integer ID that is associated with this label.
                    # The IDs are used to represent the label in ground truth images
                    # An ID of -1 means that this label does not have an ID and thus
                    # is ignored when creating ground truth images (e.g. license plate).
                    # Do not modify these IDs, since exactly these IDs are expected by the
                    # evaluation server.

    'trainId'     , # Feel free to modify these IDs as suitable for your method. Then create
                    # ground truth images with train IDs, using the tools provided in the
                    # 'preparation' folder. However, make sure to validate or submit results
                    # to our evaluation server using the regular IDs above!
                    # For trainIds, multiple labels might have the same ID. Then, these labels
                    # are mapped to the same class in the ground truth images. For the inverse
                    # mapping, we use the label that is defined first in the list below.
                    # For example, mapping all void-type classes to the same ID in training,
                    # might make sense for some approaches.
                    # Max value is 255!

    'category'    , # The name of the category that this label belongs to

    'categoryId'  , # The ID of this category. Used to create ground truth images
                    # on category level.

    'hasInstances', # Whether this label distinguishes between single instances or not

    'ignoreInEval', # Whether pixels having this class as ground truth label are ignored
                    # during evaluations or not

    'color'       , # The color of this label
    ] )

class kitti_instance(imdb):
  def __init__(self, image_set, data_path, mc):
    imdb.__init__(self, 'toy_car_'+image_set, mc)
    self._image_set = image_set
    self._data_root_path = data_path
    self._image_path = os.path.join(self._data_root_path, 'training', 'image_2')#KITTI
    self._label_path = os.path.join(self._data_root_path, 'training', 'instance')#KITTI
    self._classes = self.mc.CLASS_NAMES
    self._class_to_idx = dict(zip(self.classes, range(self.num_classes)))

    # a list of string indices of images in the directory
    self._image_idx = self._load_image_set_idx() 
    # a dict of image_idx -> [[cx, cy, w, h, cls_idx]]. x,y,w,h are not divided by
    # the image width and height

    self.labels = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label(  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'ego vehicle'          ,  1 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'rectification border' ,  2 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'out of roi'           ,  3 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'static'               ,  4 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'dynamic'              ,  5 ,      255 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
    Label(  'ground'               ,  6 ,      255 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
    Label(  'road'                 ,  7 ,        0 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
    Label(  'sidewalk'             ,  8 ,        1 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
    Label(  'parking'              ,  9 ,      255 , 'flat'            , 1       , False        , True         , (250,170,160) ),
    Label(  'rail track'           , 10 ,      255 , 'flat'            , 1       , False        , True         , (230,150,140) ),
    Label(  'building'             , 11 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
    Label(  'wall'                 , 12 ,        3 , 'construction'    , 2       , False        , False        , (102,102,156) ),
    Label(  'fence'                , 13 ,        4 , 'construction'    , 2       , False        , False        , (190,153,153) ),
    Label(  'guard rail'           , 14 ,      255 , 'construction'    , 2       , False        , True         , (180,165,180) ),
    Label(  'bridge'               , 15 ,      255 , 'construction'    , 2       , False        , True         , (150,100,100) ),
    Label(  'tunnel'               , 16 ,      255 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
    Label(  'pole'                 , 17 ,        5 , 'object'          , 3       , False        , False        , (153,153,153) ),
    Label(  'polegroup'            , 18 ,      255 , 'object'          , 3       , False        , True         , (153,153,153) ),
    Label(  'traffic light'        , 19 ,        6 , 'object'          , 3       , False        , False        , (250,170, 30) ),
    Label(  'traffic sign'         , 20 ,        7 , 'object'          , 3       , False        , False        , (220,220,  0) ),
    Label(  'vegetation'           , 21 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
    Label(  'terrain'              , 22 ,        9 , 'nature'          , 4       , False        , False        , (152,251,152) ),
    Label(  'sky'                  , 23 ,       10 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
    Label(  'person'               , 24 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
    Label(  'rider'                , 25 ,       12 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
    Label(  'car'                  , 26 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
    Label(  'truck'                , 27 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
    Label(  'bus'                  , 28 ,       15 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
    Label(  'caravan'              , 29 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ),
    Label(  'trailer'              , 30 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),
    Label(  'train'                , 31 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
    Label(  'motorcycle'           , 32 ,       17 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
    Label(  'bicycle'              , 33 ,       18 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
    Label(  'license plate'        , -1 ,       -1 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
    ]

    # name to label object
    self.name2label      = { label.name    : label for label in self.labels           }
    # id to label object
    self.id2label        = { label.id      : label for label in self.labels           }
    # trainId to label object
    self.trainId2label   = { label.trainId : label for label in reversed(self.labels) }

    self.permitted_classes = ['person', 'rider', 'car', 'truck', 'bus', 'caravan', 'trailer', 'train', 'motorcycle', 'bicycle']
    # self._rois = self._load_kitti_instance_annotation() #KITTI
    self._rois, self._poly = self._load_cityscape_8_point_annotation() #CITYSCAPE

    ## batch reader ##
    self._perm_idx = None
    self._cur_idx = 0
    # TODO(bichen): add a random seed as parameter
    self._shuffle_image_idx()

    self._eval_tool = None

  def getSingleInstanceName(self, idval):
      label = self.id2label[idval]
      # test if the new name denotes a label that actually has instances
      if not label.hasInstances:
          return None
      # all good then
      return label.name

  def assureSingleInstance(self, name):
    # if the name is known, it is not a group
    if name in self.name2label:
        return self.name2label[name], name
    # test if the name actually denotes a group
    if not name.endswith("group"):
        return None, None
    # remove group
    name = name[:-len("group")]
    # test if the new name exists
    if not name in self.name2label:
        return None, None
    return self.name2label[name], name

  def _load_image_set_idx(self):
    image_set_file = os.path.join(
        self._data_root_path, 'ImageSets', self._image_set+'.txt')
    assert os.path.exists(image_set_file), \
        'File does not exist: {}'.format(image_set_file)

    with open(image_set_file) as f:
      image_idx = [x.strip() for x in f.readlines()]
    return image_idx

  def _image_path_at(self, idx):
    image_path = os.path.join(self._image_path, idx+'.png')
    assert os.path.exists(image_path), \
        'Image does not exist: {}'.format(image_path)
    return image_path


  def get_intersecting_point(self, vert_hor, eq1, pt, m):
    pt_x, pt_y = pt
    c = pt_y - (m*pt_x)
    if vert_hor == "vert":
        x_cor = eq1
        y_cor = (m*x_cor)+c
    else:
        y_cor = eq1
        x_cor = (y_cor - c)/m
    
    return (x_cor, y_cor)

  def get_exteme_points(self, mask):
    rr, cc = np.where(mask != 0)
    sum_values = cc + rr
    diff_values = cc - rr
    # print(np.shape(sum_values), np.shape(rr), rr[0])
    xmin = min(cc)
    xmax = max(cc)
    ymin = min(rr)
    ymax = max(rr)
    min_sum_indices = np.where(sum_values == np.amin(sum_values))[0][0]
    pt_p_min = (cc[min_sum_indices], rr[min_sum_indices])
    max_sum_indices = np.where(sum_values == np.amax(sum_values))[0][0]
    pt_p_max = (cc[max_sum_indices], rr[max_sum_indices])
    min_diff_indices = np.where(diff_values == np.amin(diff_values))[0][0]
    pt_n_min = (cc[min_diff_indices], rr[min_diff_indices])
    max_diff_indices = np.where(diff_values == np.amax(diff_values))[0][0]
    pt_n_max = (cc[max_diff_indices], rr[max_diff_indices])
    eq1s = [xmin, xmin, ymax, ymax, xmax, xmax, ymin, ymin]
    vert_or_hors = ["vert", "vert", "hor", "hor", "vert", "vert", "hor", "hor"]
    pts = [pt_p_min, pt_n_min, pt_n_min, pt_p_max, pt_p_max, pt_n_max, pt_n_max, pt_p_min]
    ms = [-1, +1, +1, -1, -1, +1, +1, -1]
    intersecting_pts = []
    for eq1, pt, vert_hor, m in zip(eq1s, pts, vert_or_hors, ms):
        op_pt = self.get_intersecting_point(vert_hor, eq1, pt, m)
        intersecting_pts.append(op_pt)
    mask_vector = [xmin, ymin, xmax, ymax, intersecting_pts[0][1]-ymin, intersecting_pts[1][1]-ymin, intersecting_pts[2][0]-xmin, intersecting_pts[3][0]-xmin, intersecting_pts[4][1]-ymin, intersecting_pts[5][1]-ymin, intersecting_pts[6][0]-xmin, intersecting_pts[7][0]-xmin]
    return mask_vector

  def _load_kitti_instance_annotation(self):
    idx2annotation = {}
    for index in self._image_idx:
      bboxes = []
      filename = os.path.join(self._label_path, index+'.png')
      instance_semantic_gt = sp.imread(filename)
      instance_gt = instance_semantic_gt  % 256
      semantic_gt = instance_semantic_gt // 256
      instance_filter = np.zeros_like(instance_gt)
      instance_filter[np.where(instance_gt != 0)] = 255
      semantic_instance_mask = np.bitwise_and(semantic_gt, instance_filter)
      unique_classes = np.unique(semantic_instance_mask)
      for class_val in unique_classes:
          filtered_mask = np.zeros_like(instance_gt)
          if class_val == 0:
              continue
          indices = np.where(semantic_instance_mask == class_val)
          filtered_mask[indices] = instance_gt[indices]
          unique_instances = np.unique(filtered_mask)
          for instance_id in unique_instances:
              if instance_id == 0:
                  continue
              name = self.getSingleInstanceName(class_val)
              if name in self.permitted_classes:
                refined_mask = np.zeros_like(filtered_mask)
                ids = np.where(filtered_mask == instance_id)
                refined_mask[ids] = 255
                mask = np.uint8(refined_mask)
                cls = self._class_to_idx[name]
                vector = self.get_exteme_points(mask)
                xmin, ymin, xmax, ymax, of1, of2, of3, of4, of5, of6, of7, of8 = vector
                assert xmin >= 0.0 and xmin <= xmax, \
                    'Invalid bounding box 1 x-coord xmin {} or xmax {} at {}.txt' \
                        .format(xmin, xmax, index)
                assert ymin >= 0.0 and ymin <= ymax, \
                    'Invalid bounding box 1 y-coord ymin {} or ymax {} at {}.txt' \
                        .format(ymin, ymax, index)
                cx, cy, w, h, of1, of2, of3, of4, of5, of6, of7, of8 = bbox_transform_inv([xmin, ymin, xmax, ymax, of1, of2, of3, of4, of5, of6, of7, of8])
                bboxes.append([cx, cy, w, h, of1/h, of2/h, of3/w, of4/w, of5/h, of6/h, of7/w, of8/w, cls])
      idx2annotation[index] = bboxes # Assuming each image has a single object whiøch is true for toys dataset
    return idx2annotation

  def get_exteme_points_cityscape(self, polygon, height, width):
    outline = np.array(polygon)
    rrr, ccc = outline[:,1], outline[:,0]
    rr = []
    cc = []
    for r in rrr:
      if r < 0:
        r = 0
      if r > height:
        r = height
      rr.append(r)
    for c in ccc:
      if c < 0:
        c = 0
      if c > width:
        c = width
      cc.append(c)
    rr = np.array(rr)
    cc = np.array(cc)
    sum_values = cc + rr
    diff_values = cc - rr
#     print(np.shape(sum_values), np.shape(rr), rr[0])
    xmin = max(min(cc), 0)
    xmax = min(max(cc), width)
    ymin = max(min(rr), 0)
    ymax = min(max(rr), height)
    min_sum_indices = np.where(sum_values == np.amin(sum_values))[0][0]
    pt_p_min = (cc[min_sum_indices], rr[min_sum_indices])
    max_sum_indices = np.where(sum_values == np.amax(sum_values))[0][0]
    pt_p_max = (cc[max_sum_indices], rr[max_sum_indices])
    min_diff_indices = np.where(diff_values == np.amin(diff_values))[0][0]
    pt_n_min = (cc[min_diff_indices], rr[min_diff_indices])
    max_diff_indices = np.where(diff_values == np.amax(diff_values))[0][0]
    pt_n_max = (cc[max_diff_indices], rr[max_diff_indices])
    eq1s = [xmin, xmin, ymax, ymax, xmax, xmax, ymin, ymin]
    vert_or_hors = ["vert", "vert", "hor", "hor", "vert", "vert", "hor", "hor"]
    pts = [pt_p_min, pt_n_min, pt_n_min, pt_p_max, pt_p_max, pt_n_max, pt_n_max, pt_p_min]
    ms = [-1, +1, +1, -1, -1, +1, +1, -1]
    intersecting_pts = []
    for eq1, pt, vert_hor, m in zip(eq1s, pts, vert_or_hors, ms):
        op_pt = self.get_intersecting_point(vert_hor, eq1, pt, m)
        intersecting_pts.append(op_pt)
    mask_vector = [xmin, ymin, xmax, ymax, intersecting_pts[0][1]-ymin, intersecting_pts[1][1]-ymin, intersecting_pts[2][0]-xmin, intersecting_pts[3][0]-xmin, intersecting_pts[4][1]-ymin, intersecting_pts[5][1]-ymin, intersecting_pts[6][0]-xmin, intersecting_pts[7][0]-xmin]
    return mask_vector

  def _load_cityscape_instance_annotation(self):
    idx2annotation = {}
    rejected_image_ids = []
    for index in self._image_idx:
      bboxes = []
      filename = os.path.join(self._label_path, index[:-11]+'gtFine_polygons.json')
      instance_info = dict()
      with open(filename) as f:
        data_dict = json.load(f)
        imgHeight = data_dict['imgHeight']
        imgWidth = data_dict['imgWidth']
        instances = data_dict['objects']
        for instance in instances:
          class_name = instance['label']
          params, modified_name = self.assureSingleInstance(class_name)
          if params != None and params.hasInstances and modified_name in self.permitted_classes:
            polygon = np.array(instance['polygon'])
            cls = self._class_to_idx[modified_name]
            vector = self.get_exteme_points_cityscape(polygon, imgHeight, imgWidth) 
            xmin, ymin, xmax, ymax, of1, of2, of3, of4, of5, of6, of7, of8 = vector
            assert xmin >= 0.0 and xmin <= xmax, \
                'Invalid bounding box x-coord xmin {} or xmax {} at {}.txt' \
                    .format(xmin, xmax, index)
            assert ymin >= 0.0 and ymin <= ymax, \
                'Invalid bounding box y-coord ymin {} or ymax {} at {}.txt' \
                    .format(ymin, ymax, index)
            cx, cy, w, h, of1, of2, of3, of4, of5, of6, of7, of8 = bbox_transform_inv([xmin, ymin, xmax, ymax, of1, of2, of3, of4, of5, of6, of7, of8])
            assert not (of1 < 0 or of2 < 0 or of3 < 0 or of4 < 0 or of5 < 0 or of6 < 0 or of7< 0 or of8 < 0), "Error Occured "+ str(of1) +" "+ str(of2)+" "+ str(of3)+" "+ str(of4)+" "+ str(of5)+" "+ str(of6)+" "+ str(of7)+" "+ str(of8)
            bboxes.append([cx, cy, w, h, of1/h, of2/h, of3/w, of4/w, of5/h, of6/h, of7/w, of8/w, cls])
      # assert len(bboxes) !=0, "Error here empty bounding box appending"+str(bboxes)
      if len(bboxes) == 0:
        rejected_image_ids.append(index)
      else:
        idx2annotation[index] = bboxes
    for id_val in rejected_image_ids:
      self._image_idx.remove(id_val) #Assuming filenames are not repeated in the text file.
    return idx2annotation

  # 8 point mask parameterization
  def get_8_point_mask_parameterization(self, polygon, height, width):
    outline = np.array(polygon)
    rrr, ccc = outline[:,1], outline[:,0]
    rr = []
    cc = []
    for r in rrr:
      if r < 0:
        r = 0
      if r > height:
        r = height
      rr.append(r)
    for c in ccc:
      if c < 0:
        c = 0
      if c > width:
        c = width
      cc.append(c)
    rr = np.array(rr)
    cc = np.array(cc)
    sum_values = cc + rr
    diff_values = cc - rr
    xmin = max(min(cc), 0)
    xmax = min(max(cc), width)
    ymin = max(min(rr), 0)
    ymax = min(max(rr), height)
    width       = xmax - xmin
    height      = ymax - ymin
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
    # ms = [-1, +1, -1,  +1] #Slope of the tangents
    # offsets = []
    # for pt, m in zip(pts, ms):
    #   op_pt = self.get_perpendicular_distance(pt, m, center)
    #   offsets.append(op_pt)
    # mask_vector = [xmin, ymin, xmax, ymax, offsets[0], offsets[1], offsets[2], offsets[3], center_x, center_y, width, height]
    mask_vector = [xmin, ymin, xmax, ymax, pts[0], pts[1], pts[2], pts[3], center_x, center_y, width, height]
    return mask_vector

  def _load_cityscape_8_point_annotation(self):
    idx2annotation = {}
    idx2polygons = {}
    rejected_image_ids = []
    for index in self._image_idx:
      bboxes = []
      polygons = []
      filename = os.path.join(self._label_path, index[:-11]+'gtFine_polygons.json')
      instance_info = dict()
      with open(filename) as f:
        data_dict = json.load(f)
        imgHeight = data_dict['imgHeight']
        imgWidth = data_dict['imgWidth']
        instances = data_dict['objects']
        for instance in instances:
          class_name = instance['label']
          params, modified_name = self.assureSingleInstance(class_name)
          if params != None and params.hasInstances and modified_name in self.permitted_classes:
            polygon = np.array(instance['polygon'], dtype=np.float)
            cls = self._class_to_idx[modified_name]
            vector = self.get_8_point_mask_parameterization(polygon, imgHeight, imgWidth) 
            xmin, ymin, xmax, ymax, pt1, pt2, pt3, pt4, cx, cy, w, h = vector
            assert xmin >= 0.0 and xmin <= xmax, \
                'Invalid bounding box x-coord xmin {} or xmax {} at {}.txt' \
                    .format(xmin, xmax, index)
            assert ymin >= 0.0 and ymin <= ymax, \
                'Invalid bounding box y-coord ymin {} or ymax {} at {}.txt' \
                    .format(ymin, ymax, index)
            # cx, cy, w, h, of1, of2, of3, of4, of5, of6, of7, of8 = bbox_transform_inv([xmin, ymin, xmax, ymax, of1, of2, of3, of4, of5, of6, of7, of8])
            # assert not (of1 < 0 or of2 < 0 or of3 < 0 or of4 < 0), "Error Occured "+ str(of1) +" "+ str(of2)+" "+ str(of3)+" "+ str(of4)
            bboxes.append([cx, cy, w, h, cls])
            polygons.append([imgHeight, imgWidth, polygon])
      # assert len(bboxes) !=0, "Error here empty bounding box appending"+str(bboxes)
      if len(bboxes) == 0:
        rejected_image_ids.append(index)
      else:
        idx2annotation[index] = bboxes
        idx2polygons[index] = polygons
    for id_val in rejected_image_ids:
      self._image_idx.remove(id_val) #Assuming filenames are not repeated in the text file.
    return idx2annotation, idx2polygons

  def evaluate_detections(self, eval_dir, global_step, all_boxes):
    """Evaluate detection results.
    Args:
      eval_dir: directory to write evaluation logs
      global_step: step of the checkpoint
      all_boxes: all_boxes[cls][image] = N x 5 arrays of 
        [xmin, ymin, xmax, ymax, score]
    Returns:
      aps: array of average precisions.
      names: class names corresponding to each ap
    """
    det_file_dir = os.path.join(
        eval_dir, 'detection_files_{:s}'.format(global_step), 'data')
    if not os.path.isdir(det_file_dir):
      os.makedirs(det_file_dir)

    for im_idx, index in enumerate(self._image_idx):
      filename = os.path.join(det_file_dir, index+'.txt')
      with open(filename, 'wt') as f:
        for cls_idx, cls in enumerate(self._classes):
          dets = all_boxes[cls_idx][im_idx]
          for k in range(len(dets)):
            f.write(
                '{:s} -1 -1 0.0 {:.2f} {:.2f} {:.2f} {:.2f} 0.0 0.0 0.0 0.0 0.0 '
                '0.0 0.0 {:.3f}\n'.format(
                    cls.lower(), dets[k][0], dets[k][1], dets[k][2], dets[k][3],
                    dets[k][4])
            )

    cmd = self._eval_tool + ' ' \
          + os.path.join(self._data_root_path, 'training') + ' ' \
          + os.path.join(self._data_root_path, 'ImageSets',
                         self._image_set+'.txt') + ' ' \
          + os.path.dirname(det_file_dir) + ' ' + str(len(self._image_idx))

    print('Running: {}'.format(cmd))
    status = subprocess.call(cmd, shell=True)

    aps = []
    names = []
    for cls in self._classes:
      det_file_name = os.path.join(
          os.path.dirname(det_file_dir), 'stats_{:s}_ap.txt'.format(cls))
      if os.path.exists(det_file_name):
        with open(det_file_name, 'r') as f:
          lines = f.readlines()
        assert len(lines) == 3, \
            'Line number of {} should be 3'.format(det_file_name)

        aps.append(float(lines[0].split('=')[1].strip()))
        aps.append(float(lines[1].split('=')[1].strip()))
        aps.append(float(lines[2].split('=')[1].strip()))
      else:
        aps.extend([0.0, 0.0, 0.0])

      names.append(cls+'_easy')
      names.append(cls+'_medium')
      names.append(cls+'_hard')

    return aps, names

  def do_detection_analysis_in_eval(self, eval_dir, global_step):
    det_file_dir = os.path.join(
        eval_dir, 'detection_files_{:s}'.format(global_step), 'data')
    det_error_dir = os.path.join(
        eval_dir, 'detection_files_{:s}'.format(global_step),
        'error_analysis')
    if not os.path.exists(det_error_dir):
      os.makedirs(det_error_dir)
    det_error_file = os.path.join(det_error_dir, 'det_error_file.txt')

    stats = self.analyze_detections(det_file_dir, det_error_file)
    ims = self.visualize_detections(
        image_dir=self._image_path,
        image_format='.png',
        det_error_file=det_error_file,
        output_image_dir=det_error_dir,
        num_det_per_type=10
    )

    return stats, ims

  def analyze_detections(self, detection_file_dir, det_error_file):
    def _save_detection(f, idx, error_type, det, score):
      f.write(
          '{:s} {:s} {:.1f} {:.1f} {:.1f} {:.1f} {:s} {:.3f}\n'.format(
              idx, error_type,
              det[0]-det[2]/2., det[1]-det[3]/2.,
              det[0]+det[2]/2., det[1]+det[3]/2.,
              self._classes[int(det[4])], 
              score
          )
      )

    # load detections
    self._det_rois = {}
    for idx in self._image_idx:
      det_file_name = os.path.join(detection_file_dir, idx+'.txt')
      with open(det_file_name) as f:
        lines = f.readlines()
      f.close()
      bboxes = []
      for line in lines:
        obj = line.strip().split(' ')
        cls = self._class_to_idx[obj[0].lower().strip()]
        xmin = float(obj[4])
        ymin = float(obj[5])
        xmax = float(obj[6])
        ymax = float(obj[7])
        score = float(obj[-1])

        x, y, w, h = bbox_transform_inv([xmin, ymin, xmax, ymax])
        bboxes.append([x, y, w, h, cls, score])
      bboxes.sort(key=lambda x: x[-1], reverse=True)
      self._det_rois[idx] = bboxes

    # do error analysis
    num_objs = 0.
    num_dets = 0.
    num_correct = 0.
    num_loc_error = 0.
    num_cls_error = 0.
    num_bg_error = 0.
    num_repeated_error = 0.
    num_detected_obj = 0.

    with open(det_error_file, 'w') as f:
      for idx in self._image_idx:
        gt_bboxes = np.array(self._rois[idx])
        num_objs += len(gt_bboxes)
        detected = [False]*len(gt_bboxes)

        det_bboxes = self._det_rois[idx]
        if len(gt_bboxes) < 1:
          continue

        for i, det in enumerate(det_bboxes):
          if i < len(gt_bboxes):
            num_dets += 1
          ious = batch_iou(gt_bboxes[:, :4], det[:4])
          max_iou = np.max(ious)
          gt_idx = np.argmax(ious)
          if max_iou > 0.1:
            if gt_bboxes[gt_idx, 4] == det[4]:
              if max_iou >= 0.5:
                if i < len(gt_bboxes):
                  if not detected[gt_idx]:
                    num_correct += 1
                    detected[gt_idx] = True
                  else:
                    num_repeated_error += 1
              else:
                if i < len(gt_bboxes):
                  num_loc_error += 1
                  _save_detection(f, idx, 'loc', det, det[5])
            else:
              if i < len(gt_bboxes):
                num_cls_error += 1
                _save_detection(f, idx, 'cls', det, det[5])
          else:
            if i < len(gt_bboxes):
              num_bg_error += 1
              _save_detection(f, idx, 'bg', det, det[5])

        for i, gt in enumerate(gt_bboxes):
          if not detected[i]:
            _save_detection(f, idx, 'missed', gt, -1.0)
        num_detected_obj += sum(detected)
    f.close()

    print ('Detection Analysis:')
    print ('    Number of detections: {}'.format(num_dets))
    print ('    Number of objects: {}'.format(num_objs))
    print ('    Percentage of correct detections: {}'.format(
      num_correct/num_dets))
    print ('    Percentage of localization error: {}'.format(
      num_loc_error/num_dets))
    print ('    Percentage of classification error: {}'.format(
      num_cls_error/num_dets))
    print ('    Percentage of background error: {}'.format(
      num_bg_error/num_dets))
    print ('    Percentage of repeated detections: {}'.format(
      num_repeated_error/num_dets))
    print ('    Recall: {}'.format(
      num_detected_obj/num_objs))

    out = {}
    out['num of detections'] = num_dets
    out['num of objects'] = num_objs
    out['% correct detections'] = num_correct/num_dets
    out['% localization error'] = num_loc_error/num_dets
    out['% classification error'] = num_cls_error/num_dets
    out['% background error'] = num_bg_error/num_dets
    out['% repeated error'] = num_repeated_error/num_dets
    out['% recall'] = num_detected_obj/num_objs

    return out

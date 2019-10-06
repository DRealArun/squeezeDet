# Author: Bichen Wu (bichen@berkeley.edu) 08/25/2016

"""Train"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
from datetime import datetime
import os.path
import sys
import time

import numpy as np
from six.moves import xrange
import tensorflow as tf
import threading
import math
import copy

from config import *
from dataset import pascal_voc, kitti, toy_car, kitti_instance
from utils.util import sparse_to_dense, bgr_to_rgb, bbox_transform
from nets import *

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('dataset', 'CITYSCAPE',
                           """Currently only support KITTI datasets and Toy and CITYSCAPE.""")
tf.app.flags.DEFINE_string('data_path', '', """Root directory of data""")
tf.app.flags.DEFINE_string('image_set', 'train',
                           """ Can be train, trainval, val, or test""")
tf.app.flags.DEFINE_string('year', '2007',
                            """VOC challenge year. 2007 or 2012"""
                            """Only used for Pascal VOC dataset""")
tf.app.flags.DEFINE_string('train_dir', '/tmp/bichen/logs/squeezeDet/train',
                            """Directory where to write event logs """
                            """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 1000000,
                            """Maximum number of batches to run.""")
tf.app.flags.DEFINE_string('net', 'squeezeDet',
                           """Neural net architecture. """)
tf.app.flags.DEFINE_string('pretrained_model_path', '',
                           """Path to the pretrained model.""")
tf.app.flags.DEFINE_integer('summary_step', 10,
                            """Number of steps to save summary.""")
tf.app.flags.DEFINE_integer('checkpoint_step', 1000,
                            """Number of steps to save summary.""")
tf.app.flags.DEFINE_string('gpu', '0', """gpu id.""")


def _draw_box(im, box_list, label_list, color=None, cdict=None, form='center', write=False, img_name=None):
  assert form == 'center' or form == 'diagonal', \
      'bounding box format not accepted: {}.'.format(form)
  bk_im = copy.deepcopy(im)
  ht, wd, ch = np.shape(im)
  for bbox, label in zip(box_list, label_list):
    if form == 'center':
      raw_bounding_box = bbox
      bbox = bbox_transform(bbox)
    else:
      raw_bounding_box = bbox_transform_inv2(bbox)

    xmin, ymin, xmax, ymax = [int(bbox[o]) for o in range(len(bbox)) if o < 4]
    # of1, of2, of3, of4 = [bbox[o] for o in range(len(bbox)) if o >= 4]

    # w = xmax - xmin
    # h = ymax - ymin

    def get_intersecting_point_new(vert_hor, eq1, pt, m):
      pt_x, pt_y = pt
      c = pt_y - (m*pt_x)
      if vert_hor == "vert":
          x_cor = eq1
          y_cor = (m*x_cor) + c
      else:
          y_cor = eq1
          x_cor = (y_cor - c)/m
      return (x_cor, y_cor)
                 
    def decode_parameterization(mask_vector):
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
          op_pt = get_intersecting_point_new(vert_hor, eq1, pt, m)
          intersecting_pts.append(op_pt)
      return intersecting_pts

    points = decode_parameterization(raw_bounding_box)
    points = np.round(points)
    points = np.array(points, 'int32')

    l = label.split(':')[0] # text before "CLASS: (PROB)"
    if cdict and l in cdict:
      c = cdict[l]
    else:
      if color == None:
        c = (np.random.choice(256), np.random.choice(256), np.random.choice(256))
      else:
        c = color

    # draw box
    cv2.rectangle(im, (xmin, ymin), (xmax, ymax), c, 1)
    # draw label
    font = cv2.FONT_HERSHEY_SIMPLEX
    if color == None:
      color_mask = np.zeros((ht, wd, 3), np.uint8)
      cv2.fillConvexPoly(color_mask, points, c)
      # gray1 = cv2.cvtColor(color_mask, cv2.COLOR_BGR2GRAY) # convert to grayscale
      # ret, rough_mask = cv2.threshold(gray1, 126, 255, cv2.THRESH_BINARY)
      im[color_mask > 0] = bk_im[color_mask > 0]
      im[color_mask > 0] = 0.6*im[color_mask > 0]  + 0.4*color_mask[color_mask > 0]
      cv2.putText(im, label, (int(raw_bounding_box[0]), int(raw_bounding_box[1])), font, 0.3, (0,0,0), 1)
    else:
      cv2.putText(im, label, (int(raw_bounding_box[0]), int(raw_bounding_box[1])), font, 0.3, c, 1)

    for p in range(len(points)):
      cv2.line(im, tuple(points[p]), tuple(points[(p+1)%len(points)]), c)
    if write:
      trainval_file = r"C:\Users\Arun\Downloads\Deep-Learning-master\Custom_Mask_RCNN\Generated_annotations\\"+str(img_name)+".txt"
      print(trainval_file)
      with open(trainval_file, 'w') as f:
          f.write('{} {} {} {} {} {} {} {} {} {} {} {}\n'.format(xmin, ymin, xmax, ymax, of1, of2, of3, of4, of5, of6, of7, of8))

def _viz_prediction_result(model, images, bboxes, labels, batch_det_bbox,
                           batch_det_class, batch_det_prob):
  mc = model.mc
  for i in range(len(images)):
    # draw ground truth
    _draw_box(
        images[i], bboxes[i],
        [mc.CLASS_NAMES[idx] for idx in labels[i]]) #specify color if you dont want masks

    # draw prediction
    det_bbox, det_prob, det_class = model.filter_prediction(
        batch_det_bbox[i], batch_det_prob[i], batch_det_class[i])

    keep_idx    = [idx for idx in range(len(det_prob)) \
                      if det_prob[idx] > mc.PLOT_PROB_THRESH]
    det_bbox    = [det_bbox[idx] for idx in keep_idx]
    det_prob    = [det_prob[idx] for idx in keep_idx]
    det_class   = [det_class[idx] for idx in keep_idx]

    _draw_box(
        images[i], det_bbox,
        [mc.CLASS_NAMES[idx]+': (%.2f)'% prob \
            for idx, prob in zip(det_class, det_prob)],
        (0, 0, 255)) #specify color if you dont want masks


def train():
  """Train SqueezeDet model"""
  assert FLAGS.dataset == 'KITTI' or FLAGS.dataset == 'TOY' or FLAGS.dataset == 'KITTI_INSTANCE' or FLAGS.dataset == 'CITYSCAPE', \
      'Currently only support KITTI, KITTI_INSTANCE and TOY dataset'

  os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

  with tf.Graph().as_default():

    assert FLAGS.net == 'vgg16' or FLAGS.net == 'resnet50' \
        or FLAGS.net == 'squeezeDet' or FLAGS.net == 'squeezeDet+', \
        'Selected neural net architecture not supported: {}'.format(FLAGS.net)
    if FLAGS.net == 'vgg16':
      mc = kitti_vgg16_config()
      mc.IS_TRAINING = True
      mc.PRETRAINED_MODEL_PATH = FLAGS.pretrained_model_path
      model = VGG16ConvDet(mc)
    elif FLAGS.net == 'resnet50':
      mc = kitti_res50_config()
      mc.IS_TRAINING = True
      mc.PRETRAINED_MODEL_PATH = FLAGS.pretrained_model_path
      model = ResNet50ConvDet(mc)
    elif FLAGS.net == 'squeezeDet':
      mc = kitti_squeezeDet_config()
      mc.IS_TRAINING = True
      mc.PRETRAINED_MODEL_PATH = FLAGS.pretrained_model_path
      model = SqueezeDet(mc)
    elif FLAGS.net == 'squeezeDet+':
      mc = kitti_squeezeDetPlus_config()
      mc.IS_TRAINING = True
      mc.PRETRAINED_MODEL_PATH = FLAGS.pretrained_model_path
      model = SqueezeDetPlus(mc)

    # imdb = kitti(FLAGS.image_set, FLAGS.data_path, mc)
    # imdb = toy_car(FLAGS.image_set, FLAGS.data_path, mc)
    imdb = kitti_instance(FLAGS.image_set, FLAGS.data_path, mc)
    print("CLASS", imdb._class_to_idx)

    # save model size, flops, activations by layers
    with open(os.path.join(FLAGS.train_dir, 'model_metrics.txt'), 'w') as f:
      f.write('Number of parameter by layer:\n')
      count = 0
      for c in model.model_size_counter:
        f.write('\t{}: {}\n'.format(c[0], c[1]))
        count += c[1]
      f.write('\ttotal: {}\n'.format(count))

      count = 0
      f.write('\nActivation size by layer:\n')
      for c in model.activation_counter:
        f.write('\t{}: {}\n'.format(c[0], c[1]))
        count += c[1]
      f.write('\ttotal: {}\n'.format(count))

      count = 0
      f.write('\nNumber of flops by layer:\n')
      for c in model.flop_counter:
        f.write('\t{}: {}\n'.format(c[0], c[1]))
        count += c[1]
      f.write('\ttotal: {}\n'.format(count))
    f.close()
    print ('Model statistics saved to {}.'.format(
      os.path.join(FLAGS.train_dir, 'model_metrics.txt')))

    def _load_data(load_to_placeholder=True):
      # read batch input
      image_per_batch, label_per_batch, box_delta_per_batch, aidx_per_batch, \
          bbox_per_batch, strength_per_batch = imdb.read_batch()

      label_indices, bbox_indices, box_delta_values, mask_indices, box_values, strength_values, strength_indices\
          = [], [], [], [], [], [], []
      aidx_set = set()
      num_discarded_labels = 0
      num_labels = 0
      for i in range(len(label_per_batch)): # batch_size
        for j in range(len(label_per_batch[i])): # number of annotations
          num_labels += 1
          if (i, aidx_per_batch[i][j]) not in aidx_set:
            aidx_set.add((i, aidx_per_batch[i][j]))
            label_indices.append(
                [i, aidx_per_batch[i][j], label_per_batch[i][j]])
            mask_indices.append([i, aidx_per_batch[i][j]])
            bbox_indices.extend(
                [[i, aidx_per_batch[i][j], k] for k in range(8)])
            box_delta_values.extend(box_delta_per_batch[i][j])
            box_values.extend(bbox_per_batch[i][j])
            strength_values.append(strength_per_batch[i][j])
            strength_indices.append([i, aidx_per_batch[i][j]])
          else:
            num_discarded_labels += 1

      if mc.DEBUG_MODE:
        print ('Warning: Discarded {}/({}) labels that are assigned to the same '
               'anchor'.format(num_discarded_labels, num_labels))

      if load_to_placeholder:
        image_input = model.ph_image_input
        input_mask = model.ph_input_mask
        box_delta_input = model.ph_box_delta_input
        box_input = model.ph_box_input
        labels = model.ph_labels
        fg_strength = model.ph_fg_strength
      else:
        image_input = model.image_input
        input_mask = model.input_mask
        box_delta_input = model.box_delta_input
        box_input = model.box_input
        labels = model.labels
        fg_strength = model.fg_strength

      feed_dict = {
          image_input: image_per_batch,
          input_mask: np.reshape(
              sparse_to_dense(
                  mask_indices, [mc.BATCH_SIZE, mc.ANCHORS],
                  [1.0]*len(mask_indices)),
              [mc.BATCH_SIZE, mc.ANCHORS, 1]),
          box_delta_input: sparse_to_dense(
              bbox_indices, [mc.BATCH_SIZE, mc.ANCHORS, 8],
              box_delta_values),
          box_input: sparse_to_dense(
              bbox_indices, [mc.BATCH_SIZE, mc.ANCHORS, 8],
              box_values),
          labels: sparse_to_dense(
              label_indices,
              [mc.BATCH_SIZE, mc.ANCHORS, mc.CLASSES],
              [1.0]*len(label_indices)),
          fg_strength: np.reshape(sparse_to_dense(
                  strength_indices, [mc.BATCH_SIZE, mc.ANCHORS],
                  strength_values),
              [mc.BATCH_SIZE, mc.ANCHORS, 1]),
      }

      return feed_dict, image_per_batch, label_per_batch, bbox_per_batch, strength_per_batch

    def _enqueue(sess, coord):
      try:
        while not coord.should_stop():
          feed_dict, _, _, _, _ = _load_data()
          sess.run(model.enqueue_op, feed_dict=feed_dict)
          if mc.DEBUG_MODE:
            print ("added to the queue")
        if mc.DEBUG_MODE:
          print ("Finished enqueue")
      except Exception as e:
        coord.request_stop(e)

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

    saver = tf.train.Saver(tf.global_variables())
    summary_op = tf.summary.merge_all()

    ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)

    summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)

    init = tf.global_variables_initializer()
    sess.run(init)

    coord = tf.train.Coordinator()

    if mc.NUM_THREAD > 0:
      enq_threads = []
      for _ in range(mc.NUM_THREAD):
        enq_thread = threading.Thread(target=_enqueue, args=[sess, coord])
        # enq_thread.isDaemon()
        enq_thread.start()
        enq_threads.append(enq_thread)

    threads = tf.train.start_queue_runners(coord=coord, sess=sess)
    run_options = tf.RunOptions(timeout_in_ms=60000)

    # try: 
    for step in xrange(FLAGS.max_steps):
      if coord.should_stop():
        sess.run(model.FIFOQueue.close(cancel_pending_enqueues=True))
        coord.request_stop()
        coord.join(threads)
        break

      start_time = time.time()

      if step % FLAGS.summary_step == 0:
        feed_dict, image_per_batch, label_per_batch, bbox_per_batch, strength_per_batch = \
            _load_data(load_to_placeholder=False)
        op_list = [
            model.train_op, model.loss, summary_op, model.det_boxes,
            model.det_probs, model.det_class, model.conf_loss,
            model.bbox_loss, model.class_loss, model.filters, model.strength_loss
        ]
        _, loss_value, summary_str, det_boxes, det_probs, det_class, \
            conf_loss, bbox_loss, class_loss, filters, strength_loss = sess.run(
                op_list, feed_dict=feed_dict)

        print("Filter Shapes: ", np.shape(filters))
        _viz_prediction_result(
            model, image_per_batch, bbox_per_batch, label_per_batch, det_boxes,
            det_class, det_probs)
        image_per_batch = bgr_to_rgb(image_per_batch)
        viz_summary = sess.run(
            model.viz_op, feed_dict={model.image_to_show: image_per_batch})

        def visualize_features(filters):
          filter_shape = np.shape(filters)
          padded_filters = np.pad(filters, ((0, 0), (3, 3), (3, 3), (0, 0)), 'constant', constant_values=(0))
          reshaped_filters = np.reshape(padded_filters,(filter_shape[0],28*32,16*82,1)) 
          return reshaped_filters
        processed_filters = visualize_features(filters)

        viz_filters_summary = sess.run(
            model.viz_filt_op, feed_dict={model.filters_to_show: processed_filters})

        summary_writer.add_summary(summary_str, step)
        summary_writer.add_summary(viz_summary, step)
        summary_writer.add_summary(viz_filters_summary, step)
        summary_writer.flush()

        print ('conf_loss: {}, bbox_loss: {}, class_loss: {}, strength_loss:{}'.
            format(conf_loss, bbox_loss, class_loss, strength_loss))
      else:
        if mc.NUM_THREAD > 0:
          _, loss_value, conf_loss, bbox_loss, class_loss, strength_loss = sess.run(
              [model.train_op, model.loss, model.conf_loss, model.bbox_loss,
               model.class_loss, model.strength_loss], options=run_options)
        else:
          feed_dict, _, _, _ = _load_data(load_to_placeholder=False)
          _, loss_value, conf_loss, bbox_loss, class_loss, strength_loss = sess.run(
              [model.train_op, model.loss, model.conf_loss, model.bbox_loss,
               model.class_loss, model.strength_loss], feed_dict=feed_dict)

      duration = time.time() - start_time

      assert not np.isnan(loss_value), \
          'Model diverged. Total loss: {}, conf_loss: {}, bbox_loss: {}, ' \
          'class_loss: {}, strength_loss:{}'.format(loss_value, conf_loss, bbox_loss, class_loss, strength_loss)

      if step % 10 == 0:
        num_images_per_step = mc.BATCH_SIZE
        images_per_sec = num_images_per_step / duration
        sec_per_batch = float(duration)
        format_str = ('%s: step %d, loss = %.2f (%.1f images/sec; %.3f '
                      'sec/batch)')
        print (format_str % (datetime.now(), step, loss_value,
                             images_per_sec, sec_per_batch))
        sys.stdout.flush()

      # Save the model checkpoint periodically.
      if step % FLAGS.checkpoint_step == 0 or (step + 1) == FLAGS.max_steps:
        checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)
    # except Exception, e:
    #   coord.request_stop(e)
    # finally:
    #   coord.request_stop()
    #   coord.join(threads)

def main(argv=None):  # pylint: disable=unused-argument
  if tf.gfile.Exists(FLAGS.train_dir):
    tf.gfile.DeleteRecursively(FLAGS.train_dir)
  tf.gfile.MakeDirs(FLAGS.train_dir)
  train()


if __name__ == '__main__':
  tf.app.run()

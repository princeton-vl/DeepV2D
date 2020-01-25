import tensorflow as tf
import numpy as np

import os
import cv2

from core.config import cfg
from geometry.transformation import *


FLIP_OFFSET = 100000

def random_frame_idx(n, m):
    ix = [0] + np.random.choice(range(1,n), m, replace=False).tolist()
    return np.array(ix, dtype='int32')


def _read_prediction_py(id, filled):
    depth_path = os.path.join(cfg.TMP_DIR, "%d.png"%id)
    if not os.path.isfile(depth_path):
        return filled

    depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
    return (depth/5000.0).astype(np.float32)


def _write_prediction_py(ids, prediction):
    for i in range(ids.shape[0]):
        depth = (prediction[i]*5000).astype(np.uint16)
        depth_path = os.path.join(cfg.TMP_DIR, "%d.png"%ids[i])
        cv2.imwrite(depth_path, depth)

    return np.int32(1.0)


class DataLayer(object):

    def __init__(self, tfrecords_file, batch_size=2):
        self.tfrecords_file = tfrecords_file
        self.batch_size = batch_size

    def augument(self, images):
        # randomly shift gamma
        images = tf.cast(images, 'float32')
        random_gamma = tf.random_uniform([], 0.9, 1.1)
        images = 255.0*((images/255.0)**random_gamma)

        # randomly shift brightness
        random_brightness = tf.random_uniform([], 0.8, 1.2)
        images *= random_brightness

        # randomly shift color
        random_colors = tf.random_uniform([3], 0.8, 1.2)
        images *= tf.reshape(random_colors, [1, 1, 1, 3])

        images = tf.clip_by_value(images, 0.0, 255.0)
        images = tf.cast(images, 'uint8')

        return images


    def scale(self, id, images, poses, depth_gt, filled, intrinsics):
        """ Random scale augumentation """
        
        if len(cfg.INPUT.SCALES) > 1:
            scales = tf.constant(cfg.INPUT.SCALES)
            scale_ix = tf.random.uniform([], 0, len(cfg.INPUT.SCALES), dtype=tf.int32)

            s = tf.gather(scales, scale_ix)
            ht = cfg.INPUT.HEIGHT
            wd = cfg.INPUT.WIDTH
            ht1 = tf.cast(ht * s, tf.int32)
            wd1 = tf.cast(wd * s, tf.int32)

            dx = (wd1 - wd) // 2 
            dy = (ht1 - ht) // 2

            images = tf.image.resize_bilinear(images, [ht1, wd1])[:, dy:dy+ht, dx:dx+wd]
            depth_gt = tf.image.resize_nearest_neighbor(depth_gt[tf.newaxis], [ht1, wd1])[:, dy:dy+ht, dx:dx+wd]
            filled = tf.image.resize_nearest_neighbor(filled[tf.newaxis], [ht1, wd1])[:, dy:dy+ht, dx:dx+wd]

            images = tf.reshape(images,  [cfg.INPUT.SAMPLES+1, ht, wd, 3])
            depth_gt = tf.reshape(depth_gt, [ht, wd, 1])
            filled = tf.reshape(filled, [ht, wd, 1])

            intrinsics = (intrinsics * s) - [0, 0, dx, dy]
            id = id + tf.constant(FLIP_OFFSET) * scale_ix

        return id, images, poses, depth_gt, filled, intrinsics


    def flip(self, id, images, poses, depth_gt, filled, intrinsics):

        do_flip = tf.random_uniform([], 0, 1) < 0.5
        images = tf.cond(do_flip, lambda: images, lambda: images[:, :, ::-1])
        depth_gt = tf.cond(do_flip, lambda: depth_gt, lambda: depth_gt[:, ::-1])
        filled = tf.cond(do_flip, lambda: filled, lambda: filled[:, ::-1])

        wd = tf.to_float(tf.shape(images)[2])
        fx, fy, cx, cy = tf.unstack(intrinsics, num=4, axis=-1)
        intrinsics_flipped = tf.stack([-fx, fy, wd - cx, cy], axis=-1)
        intrinsics = tf.cond(do_flip, lambda: intrinsics, lambda: intrinsics_flipped)

        flip_offset = tf.constant(FLIP_OFFSET, dtype=tf.int32)
        id = tf.cond(do_flip, lambda: id, lambda: id + flip_offset)

        return id, images, poses, depth_gt, filled, intrinsics


    def read_example(self, tfrecord_serialized):

        tfrecord_features = tf.parse_single_example(tfrecord_serialized,
            features={
                'id': tf.FixedLenFeature([], tf.string),
                'dim': tf.FixedLenFeature([], tf.string),
                'images': tf.FixedLenFeature([], tf.string),
                'poses': tf.FixedLenFeature([], tf.string),
                'depth': tf.FixedLenFeature([], tf.string),
                'filled': tf.FixedLenFeature([], tf.string),
                'intrinsics': tf.FixedLenFeature([], tf.string),
            }, name='features')

        id = tf.decode_raw(tfrecord_features['id'], tf.int32)
        dim = tf.decode_raw(tfrecord_features['dim'], tf.int32)

        images = tf.decode_raw(tfrecord_features['images'], tf.uint8)
        poses = tf.decode_raw(tfrecord_features['poses'], tf.float32)
        depth = tf.decode_raw(tfrecord_features['depth'], tf.float32)
        filled = tf.decode_raw(tfrecord_features['filled'], tf.float32)
        intrinsics = tf.decode_raw(tfrecord_features['intrinsics'], tf.float32)

        id = tf.reshape(id, [])
        dim = tf.reshape(dim, [4])

        frames = cfg.INPUT.FRAMES
        height = cfg.INPUT.HEIGHT
        width = cfg.INPUT.WIDTH

        images = tf.reshape(images, [frames, height, width, 3])
        poses = tf.reshape(poses, [frames, 4, 4])
        depth = tf.reshape(depth, [height, width, 1])
        filled = tf.reshape(filled, [height, width, 1])
        intrinsics = tf.reshape(intrinsics, [4])

        # randomly sample from neighboring frames (used for NYU)
        if cfg.INPUT.SAMPLES > 0:
            ix = tf.py_func(random_frame_idx, [cfg.INPUT.FRAMES, cfg.INPUT.SAMPLES], tf.int32)
            ix = tf.reshape(ix, [cfg.INPUT.SAMPLES+1])

            images = tf.gather(images, ix, axis=0)
            poses = tf.gather(poses, ix, axis=0)

        do_augument = tf.random_uniform([], 0, 1)
        images = tf.cond(do_augument<0.5, lambda: self.augument(images), lambda: images)

        id, images, poses, depth, filled, intrinsics = \
            self.scale(id, images, poses, depth, filled, intrinsics)

        prediction = tf.py_func(_read_prediction_py, [id, filled], tf.float32)
        prediction = tf.reshape(prediction, [height, width, 1])

        return id, images, poses, depth, filled, prediction, intrinsics


    def next(self):

        filename_queue = tf.train.string_input_producer([self.tfrecords_file])
        tfreader = tf.TFRecordReader()

        _, serialized = tfreader.read(filename_queue)

        example = self.read_example(serialized)
        id, images, poses, depth_gt, filled, depth_pred, intrinsics \
            = tf.train.batch(example, batch_size=self.batch_size, num_threads=2, capacity=64)

        images = tf.cast(images, tf.float32)
        return id, images, poses, depth_gt, filled, depth_pred, intrinsics


    def write(self, id, prediction):
        return tf.py_func(_write_prediction_py, [id, prediction], tf.int32)



def scale(id, images, poses, depth_gt, filled, pred, intrinsics):
    """ Random scale augumentation """

    if len(cfg.INPUT.SCALES) > 1:
        scales = tf.constant(cfg.INPUT.SCALES)
        scale_ix = tf.random.uniform([], 0, len(cfg.INPUT.SCALES), dtype=tf.int32)

        s = tf.gather(scales, scale_ix)
        ht = cfg.INPUT.HEIGHT
        wd = cfg.INPUT.WIDTH

        ht1 = tf.cast(ht * s, tf.int32)
        wd1 = tf.cast(wd * s, tf.int32)

        dx = (wd1 - wd) // 2 
        dy = (ht1 - ht) // 2

        depth_gt = tf.reshape(depth_gt, [1, ht, wd, 1])
        filled = tf.reshape(filled, [1, ht, wd, 1])

        images = tf.image.resize_bilinear(images, [ht1, wd1])[:, dy:dy+ht, dx:dx+wd]
        depth_gt = tf.image.resize_nearest_neighbor(depth_gt, [ht1, wd1])[:, dy:dy+ht, dx:dx+wd]
        filled = tf.image.resize_nearest_neighbor(filled, [ht1, wd1])[:, dy:dy+ht, dx:dx+wd]

        images = tf.reshape(images,  [4, ht, wd, 3])
        depth_gt = tf.reshape(depth_gt, [ht, wd, 1])
        filled = tf.reshape(filled, [ht, wd, 1])
        pred = tf.reshape(pred, [ht, wd, 1])

        intrinsics = (intrinsics * s) - [0, 0, dx, dy]
        id = id + tf.constant(FLIP_OFFSET) * scale_ix

    return id, images, poses, depth_gt, filled, pred, intrinsics



def augument(images):
    # randomly shift gamma
    images = tf.cast(images, 'float32')
    random_gamma = tf.random_uniform([], 0.9, 1.1)
    images = 255.0*((images/255.0)**random_gamma)

    # randomly shift brightness
    random_brightness = tf.random_uniform([], 0.8, 1.2)
    images *= random_brightness

    # randomly shift color
    random_colors = tf.random_uniform([3], 0.8, 1.2)
    images *= tf.reshape(random_colors, [1, 1, 1, 3])

    images = tf.clip_by_value(images, 0.0, 255.0)
    images = tf.cast(images, 'uint8')

    return images


def prepare_inputs(images, poses, depth, filled, pred, intrinsics, ids):
    images = augument(images)

    ids, images, poses, depth, filled, pred, intrinsics = \
        scale(ids, images, poses, depth, filled, pred, intrinsics)

    return images, poses, depth, filled, pred, intrinsics, ids


class DBDataLayer:
    def __init__(self, db, batch_size=1, augument=False):
        self.db = db
        self.batch_size = batch_size
        training_generator = iter(self.db)
        generator_data_type = (tf.uint8, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.int32)
        training_set = tf.data.Dataset.from_generator(lambda: training_generator, generator_data_type)
        training_set = training_set.map(prepare_inputs)
        
        training_set = training_set.prefetch(buffer_size=20)
        training_set = training_set.batch(batch_size)
        self.training_iterator = training_set.make_initializable_iterator()
        
    def next(self):
        frames, height, width = self.db.shape()
        images, poses, depth, filled, pred, intrinsics, ids = self.training_iterator.get_next()
        
        images.set_shape(tf.TensorShape([self.batch_size, frames, height, width, 3]))
        poses.set_shape(tf.TensorShape([self.batch_size, frames, 4, 4]))
        depth.set_shape(tf.TensorShape([self.batch_size, height, width, 1]))
        filled.set_shape(tf.TensorShape([self.batch_size, height, width, 1]))
        pred.set_shape(tf.TensorShape([self.batch_size, height, width, 1]))
        intrinsics.set_shape(tf.TensorShape([self.batch_size, 4]))
        ids.set_shape(tf.TensorShape([self.batch_size]))

        images = tf.cast(images, tf.float32)
        return ids, images, poses, depth, filled, pred, intrinsics

    def init(self, sess):
        sess.run(self.training_iterator.initializer)

    def write(self, id, prediction):
        return tf.py_func(_write_prediction_py, [id, prediction], tf.int32)
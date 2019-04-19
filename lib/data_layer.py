import tensorflow as tf
import numpy as np

import os
import cv2

from core.config import cfg



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

        prediction = tf.py_func(_read_prediction_py, [id, filled], tf.float32)
        prediction = tf.reshape(prediction, [height, width, 1])

        return id, images, poses, depth, filled, prediction, intrinsics


    def next(self):

        filename_queue = tf.train.string_input_producer([self.tfrecords_file])
        tfreader = tf.TFRecordReader()

        _, serialized = tfreader.read(filename_queue)

        example = self.read_example(serialized)
        batch = tf.train.batch(example, batch_size=self.batch_size, num_threads=2, capacity=64)

        return batch


    def write(self, id, prediction):
        return tf.py_func(_write_prediction_py, [id, prediction], tf.int32)

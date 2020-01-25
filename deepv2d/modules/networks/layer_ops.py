import tensorflow as tf
import numpy as np
slim = tf.contrib.slim

from core.config import cfg


def bnrelu(x):
    return tf.nn.relu(slim.batch_norm(x))


def conv3d(x, dim, stride=1, bn=True):
    if bn:
        return slim.conv3d(bnrelu(x), dim, [3, 3, 3], stride=stride)
    else:
        return slim.conv3d(tf.nn.relu(x), dim, [3, 3, 3], stride=stride)

def conv2d(x, dim, stride=1, bn=True):
    if bn:
        return slim.conv2d(bnrelu(x), dim, [3, 3], stride=stride)
    else:
        return slim.conv2d(tf.nn.relu(x), dim, [3, 3], stride=stride)

def res_conv2d(x, dim, stride=1):
    if stride==1:
        y = conv2d(conv2d(x, dim), dim)
    else:
        y = conv2d(conv2d(x, dim), dim, stride=2)
        x = slim.conv2d(x, dim, [1,1], stride=2)

    out = x + y
    tf.add_to_collection("checkpoints", out)

    return out

def upnn3d(x, y, sc=2):
    dim = x.get_shape().as_list()[-1]
    bx, hx, wx, dx, _ = tf.unstack(tf.shape(x), num=5)
    by, hy, wy, dy, _ = tf.unstack(tf.shape(y), num=5)

    x1 = tf.reshape(tf.tile(x, [1,1,sc,sc,sc]), [bx, sc*hx, sc*wx, sc*dx, dim])
    if not (sc*hx==hy and sc*wx==wy):
        x1 = x1[:, :hy, :wy]

    return x1

def upnn2d(x, y, sc=2):
    dim = x.get_shape().as_list()[-1]
    bx, hx, wx, _ = tf.unstack(tf.shape(x), num=4)
    by, hy, wy, _ = tf.unstack(tf.shape(y), num=4)

    x1 = tf.reshape(tf.tile(x, [1,1,sc,sc]), [bx, sc*hx, sc*wx, dim])
    if not (sc*hx==hy and sc*wx==wy):
        x1 = x1[:, :hy, :wy]

    return x1

def resize_like(inputs, ref):
    iH, iW = inputs.get_shape()[1], inputs.get_shape()[2]
    rH, rW = ref.get_shape()[1], ref.get_shape()[2]
    if iH == rH and iW == rW:
        return inputs
    return tf.image.resize_nearest_neighbor(inputs, [rH.value, rW.value])

def resize_depth(depth, dim, min_depth=0):
    depth = tf.image.resize_nearest_neighbor(depth[...,tf.newaxis], dim)
    if min_depth > 0:
        depth = tf.where(depth<min_depth, min_depth*tf.ones_like(depth), depth)

    return tf.squeeze(depth, axis=-1)

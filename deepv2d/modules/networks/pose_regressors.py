

import tensorflow as tf
import numpy as np
slim = tf.contrib.slim

from special_ops.operators import clip_nan_gradients

"""
Euler rotation code taken from SfMLearner: https://github.com/tinghuiz/SfMLearner
"""

def euler2mat(z, y, x):
    # https://github.com/tinghuiz/SfMLearner
    """Converts euler angles to rotation matrix
    TODO: remove the dimension for 'N' (deprecated for converting all source
         poses altogether)
    Reference: https://github.com/pulkitag/pycaffe-utils/blob/master/rot_utils.py#L174
    Args:
      z: rotation angle along z axis (in radians) -- size = [B, N]
      y: rotation angle along y axis (in radians) -- size = [B, N]
      x: rotation angle along x axis (in radians) -- size = [B, N]
    Returns:
      Rotation matrix corresponding to the euler angles -- size = [B, N, 3, 3]
    """
    B = tf.shape(z)[0]
    N = 1
    z = tf.clip_by_value(z, -np.pi, np.pi)
    y = tf.clip_by_value(y, -np.pi, np.pi)
    x = tf.clip_by_value(x, -np.pi, np.pi)

    # Expand to B x N x 1 x 1
    z = tf.expand_dims(tf.expand_dims(z, -1), -1)
    y = tf.expand_dims(tf.expand_dims(y, -1), -1)
    x = tf.expand_dims(tf.expand_dims(x, -1), -1)

    zeros = tf.zeros([B, N, 1, 1])
    ones  = tf.ones([B, N, 1, 1])

    cosz = tf.cos(z)
    sinz = tf.sin(z)
    rotz_1 = tf.concat([cosz, -sinz, zeros], axis=3)
    rotz_2 = tf.concat([sinz,  cosz, zeros], axis=3)
    rotz_3 = tf.concat([zeros, zeros, ones], axis=3)
    zmat = tf.concat([rotz_1, rotz_2, rotz_3], axis=2)

    cosy = tf.cos(y)
    siny = tf.sin(y)
    roty_1 = tf.concat([cosy, zeros, siny], axis=3)
    roty_2 = tf.concat([zeros, ones, zeros], axis=3)
    roty_3 = tf.concat([-siny,zeros, cosy], axis=3)
    ymat = tf.concat([roty_1, roty_2, roty_3], axis=2)

    cosx = tf.cos(x)
    sinx = tf.sin(x)
    rotx_1 = tf.concat([ones, zeros, zeros], axis=3)
    rotx_2 = tf.concat([zeros, cosx, -sinx], axis=3)
    rotx_3 = tf.concat([zeros, sinx, cosx], axis=3)
    xmat = tf.concat([rotx_1, rotx_2, rotx_3], axis=2)

    rotMat = tf.matmul(tf.matmul(xmat, ymat), zmat)
    return rotMat


def pose_vec2mat(vec, use_filler=True):
    # https://github.com/tinghuiz/SfMLearner
    """Converts 6DoF parameters to transformation matrix
    Args:
      vec: 6DoF parameters in the order of tx, ty, tz, rx, ry, rz -- [B, 6]
    Returns:
      A transformation matrix -- [B, 4, 4]
    """
    # batch_size, _ = vec.get_shape().as_list()
    batch_size = tf.shape(vec)[0]
    translation = tf.slice(vec, [0, 0], [-1, 3])
    translation = tf.expand_dims(translation, -1)
    rx = tf.slice(vec, [0, 3], [-1, 1])
    ry = tf.slice(vec, [0, 4], [-1, 1])
    rz = tf.slice(vec, [0, 5], [-1, 1])
    rot_mat = euler2mat(rz, ry, rx)
    rot_mat = tf.squeeze(rot_mat, axis=[1])
    filler = tf.constant([0.0, 0.0, 0.0, 1.0], shape=[1, 1, 4])
    filler = tf.tile(filler, [batch_size, 1, 1])
    transform_mat = tf.concat([rot_mat, translation], axis=2)
    if use_filler:
      transform_mat = tf.concat([transform_mat, filler], axis=1)
    return transform_mat


def pose_regressor_stack(image_star, images, sc=2):

    batch, frames, height, width, _ = images.get_shape().as_list()
    image_star = tf.expand_dims(image_star, 1)
    images = tf.concat([image_star, images], axis=1)

    images = tf.transpose(images, [0, 2, 3, 1, 4])
    images = tf.reshape(images, [batch, height, width, 3*(frames+1)])

    with tf.device('/cpu:0'):
        inputs = tf.image.resize_area(images, [height//sc, width//sc])

    with tf.variable_scope('pose') as sc:
        with slim.arg_scope([slim.conv2d],
                            weights_regularizer=slim.l2_regularizer(0.00005),
                            normalizer_fn=None,
                            activation_fn=tf.nn.relu):

            net = slim.conv2d(inputs, 32, [7, 7], stride=2)
            tf.add_to_collection("checkpoints", net)

            net = slim.conv2d(net, 32, [1, 7], stride=1)
            net = slim.conv2d(net, 32, [7, 1], stride=1)
            net = slim.conv2d(net, 64, [3, 3], stride=2)
            tf.add_to_collection("checkpoints", net)

            net = slim.conv2d(net, 64, [1, 5], stride=1)
            net = slim.conv2d(net, 64, [5, 1], stride=1)
            net = slim.conv2d(net, 128, [3, 3], stride=2)
            tf.add_to_collection("checkpoints", net)

            net = slim.conv2d(net, 128, [3, 3], stride=1)
            net = slim.conv2d(net, 256, [3, 3], stride=2)
            tf.add_to_collection("checkpoints", net)

            net = slim.conv2d(net, 256, [3, 3], stride=1)
            net = slim.conv2d(net, 256, [3, 3], stride=2)
            tf.add_to_collection("checkpoints", net)

            net = slim.conv2d(net, 256, [3, 3], stride=1)
            net = slim.conv2d(net, 256, [3, 3], stride=2)
            tf.add_to_collection("checkpoints", net)

            dims = net.get_shape().as_list()
            net = slim.conv2d(net, 512, [dims[1], dims[2]], padding='VALID')
            pose_vec = slim.conv2d(net, 6*frames, [1, 1], stride=1, activation_fn=None)
            pose_vec = 0.01*tf.reshape(pose_vec, [batch*frames, 6])
            pose_vec = clip_nan_gradients(pose_vec)

            tf.add_to_collection("checkpoints", pose_vec)

            # tf.summary.histogram("pose_vec", pose_vec)

            G = pose_vec2mat(pose_vec)
            return G


def pose_regressor_indv(image1, image2, sc=2):

    ht = image1.get_shape().as_list()[1]
    wd = image2.get_shape().as_list()[2]
    inputs = tf.concat([image1, image2], axis=-1)

    with tf.device('/cpu:0'):
        inputs = tf.image.resize_area(inputs, [ht//sc, wd//sc])

    with tf.variable_scope('pose') as sc:
        with slim.arg_scope([slim.conv2d],
                            weights_regularizer=slim.l2_regularizer(0.00005),
                            normalizer_fn=None,
                            activation_fn=tf.nn.relu):

            net = slim.conv2d(inputs, 32, [7, 7], stride=2)
            tf.add_to_collection("checkpoints", net)

            net = slim.conv2d(net, 32, [1, 7], stride=1)
            net = slim.conv2d(net, 32, [7, 1], stride=1)
            net = slim.conv2d(net, 64, [3, 3], stride=2)
            tf.add_to_collection("checkpoints", net)

            net = slim.conv2d(net, 64, [1, 5], stride=1)
            net = slim.conv2d(net, 64, [5, 1], stride=1)
            net = slim.conv2d(net, 128, [3, 3], stride=2)
            tf.add_to_collection("checkpoints", net)

            net = slim.conv2d(net, 128, [3, 3], stride=1)
            net = slim.conv2d(net, 256, [3, 3], stride=2)
            tf.add_to_collection("checkpoints", net)

            net = slim.conv2d(net, 256, [3, 3], stride=1)
            net = slim.conv2d(net, 256, [3, 3], stride=2)
            tf.add_to_collection("checkpoints", net)

            net = slim.conv2d(net, 256, [3, 3], stride=1)
            net = slim.conv2d(net, 256, [3, 3], stride=2)
            tf.add_to_collection("checkpoints", net)

            dims = net.get_shape().as_list()
            net = slim.conv2d(net, 512, [dims[1], dims[2]], padding='VALID')
            pose_vec = slim.conv2d(net, 6, [1, 1], stride=1, activation_fn=None)
            
            pose_vec = 0.01*tf.reshape(pose_vec, [-1, 6])
            pose_vec = clip_nan_gradients(pose_vec)
            tf.add_to_collection("checkpoints", pose_vec)    
            # tf.summary.histogram("pose_vec", pose_vec)

            G = pose_vec2mat(pose_vec)
            return G


def pose_regressor_factory(image_star, images, cfg):

    if cfg.STACK_FRAMES:
        G = pose_regressor_stack(image_star, images)
    
    else:
        batch, frames, ht, wd = [tf.shape(images)[i] for i in range(4)]
        ht = images.get_shape().as_list()[2]
        wd = images.get_shape().as_list()[3]
        image_dims = [batch*frames, ht, wd, 3]

        image_star = tf.expand_dims(image_star, 1)
        image1 = tf.tile(image_star, [1, frames, 1, 1, 1])
        image1 = tf.reshape(image1, image_dims)
        image2 = tf.reshape(images, image_dims)

        G = pose_regressor_indv(image1, image2)

    return G

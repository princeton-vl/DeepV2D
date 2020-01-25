import tensorflow as tf
import numpy as np
from utils.einsum import einsum

def intrinsics_vec_to_matrix(kvec):
    fx, fy, cx, cy = tf.unstack(kvec, num=4, axis=-1)
    z = tf.zeros_like(fx)
    o = tf.ones_like(fx)

    K = tf.stack([fx, z, cx, z, fy, cy, z, z, o], axis=-1)
    K = tf.reshape(K, kvec.get_shape().as_list()[:-1] + [3,3])
    return K

def intrinsics_matrix_to_vec(kmat):
    fx = kmat[..., 0, 0]
    fy = kmat[..., 1, 1]
    cx = kmat[..., 0, 2]
    cy = kmat[..., 1, 2]
    return tf.stack([fx, fy, cx, cy], axis=-1)

def update_intrinsics(intrinsics, delta_focal):
    kvec = intrinsics_matrix_to_vec(intrinsics)
    fx, fy, cx, cy = tf.unstack(kvec, num=4, axis=-1)
    df = tf.squeeze(delta_focal, -1)

    # update the focal lengths
    fx = tf.exp(df) * fx
    fy = tf.exp(df) * fy

    kvec = tf.stack([fx, fy, cx, cy], axis=-1)
    kmat = intrinsics_vec_to_matrix(kvec)
    return kmat

def rescale_depth(depth, downscale=4):
    depth = tf.expand_dims(depth, axis=-1)
    new_shape = tf.shape(depth)[1:3] // downscale
    depth = tf.image.resize_nearest_neighbor(depth, new_shape)
    return tf.squeeze(depth, axis=-1)

def rescale_depth_and_intrinsics(depth, intrinsics, downscale=4):
    sc = tf.constant([1.0/downscale, 1.0/downscale, 1.0], dtype=tf.float32)
    intrinsics = einsum('...ij,i->...ij', intrinsics, sc)
    depth = rescale_depth(depth, downscale=downscale)
    return depth, intrinsics

def rescale_depths_and_intrinsics(depth, intrinsics, downscale=4):
    batch, frames, height, width = [tf.shape(depth)[i] for i in range(4)]
    depth = tf.reshape(depth, [batch*frames, height, width])
    depth, intrinsics = rescale_depth_and_intrinsics(depth, intrinsics, downscale)
    depth = tf.reshape(depth,
        tf.concat(([batch, frames], tf.shape(depth)[1:]), axis=0))
    return depth, intrinsics

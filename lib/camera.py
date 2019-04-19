import tensorflow as tf
import numpy as np


def coords_grid(batch, height, width, homogeneous=False):

    cam_coords = tf.meshgrid(tf.range(width), tf.range(height))
    coords = tf.stack(cam_coords, axis=-1)
    coords = tf.to_float(tf.expand_dims(coords, 0))

    if homogeneous:
        coords = tf.concat([coords, tf.ones([1,height,width,1])], axis=-1)

    coords = tf.tile(coords, [batch, 1, 1, 1])
    return coords


def proj(X, kv, min_depth=0.01):
    """
    Project 3d point cloud to image
    Inputs:
        X: [B, H, W, 4], point cloud
        kv: [B, 4], intrinsics vector
    Outputs:
        x: [B, H, W, 2], projected poinits
    """
    fx, fy, cx, cy = tf.split(kv, [1, 1, 1, 1], axis=-1)
    ndim = len(X.get_shape().as_list())
    fx = tf.reshape(fx, [-1]+[1]*(ndim-2))
    fy = tf.reshape(fy, [-1]+[1]*(ndim-2))
    cx = tf.reshape(cx, [-1]+[1]*(ndim-2))
    cy = tf.reshape(cy, [-1]+[1]*(ndim-2))

    z = X[...,2]
    too_close = tf.abs(z) < min_depth
    z = tf.where(too_close, min_depth*tf.ones_like(z), z)

    x_px = fx*(X[...,0]/z) + cx
    y_px = fy*(X[...,1]/z) + cy

    x = tf.stack([x_px, y_px], axis=-1)
    return x


def iproj(x, z, kv):
    """
    Creates 3d point cloud from pixel coords and depth
    Inputs:
        x: [B, H, W, 2], pixel coords
        kv: [B, 4], intrinsics vector
    Outputs:
        X: [B, H, W, 4], 3d point cloud
    """
    fx, fy, cx, cy = tf.split(kv, [1, 1, 1, 1], axis=-1)
    ndim = len(x.get_shape().as_list())
    fx = tf.reshape(fx, [-1]+[1]*(ndim-2))
    fy = tf.reshape(fy, [-1]+[1]*(ndim-2))
    cx = tf.reshape(cx, [-1]+[1]*(ndim-2))
    cy = tf.reshape(cy, [-1]+[1]*(ndim-2))

    X = z * (x[...,0] - cx) / fx
    Y = z * (x[...,1] - cy) / fy

    X = tf.stack([X, Y, z, tf.ones_like(z)], axis=-1)
    return X


def point_cloud_from_depth(depth, kv):
    """ Backproject depth to point cloud"""
    batch, height, width = depth.get_shape().as_list()
    pix = coords_grid(batch, height, width)
    X = iproj(pix, depth, kv)
    return X


def camera_transform_project(G, depth, intrinsics):
    """
    Maps points from camera (I) to camera (G)

    Inputs:
        G: [B, 4, 4], SE3 transformation, camera motion
        depth: [B, H, W]
        intrinsics: [B, 4], (fx, fy, cx, cy)
    Outputs:
        x1: transformed points
    """
    X = point_cloud_from_depth(depth, intrinsics)
    X1 = tf.einsum('aij,apqj->apqi', G, X)
    x1 = proj(X1, intrinsics)

    return x1

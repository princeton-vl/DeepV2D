import tensorflow as tf
import numpy as np
slim = tf.contrib.slim

from .layer_ops import *
    

def hourglass_2d(x, n, dim, expand=64):
    dim2 = dim + expand
    x = x + conv2d(conv2d(x, dim), dim)

    pool1 = slim.max_pool2d(x, [2, 2], padding='SAME')

    low1 = conv2d(pool1, dim2)
    if n>1:
        low2 = hourglass_2d(low1, n-1, dim2)
    else:
        low2 = conv2d(low1, dim2)

    low3 = conv2d(low2, dim)
    up2 = upnn2d(low3, x)

    out = up2 + x
    tf.add_to_collection("checkpoints", out)

    return out


def hourglass_3d(x, n, dim, expand=48):
    dim2 = dim + expand

    x = x + conv3d(conv3d(x, dim), dim)
    tf.add_to_collection("checkpoints", x)

    pool1 = slim.max_pool3d(x, [2, 2, 2], padding='SAME')

    low1 = conv3d(pool1, dim2)
    if n>1:
        low2 = hourglass_3d(low1, n-1, dim2)
    else:
        low2 = low1 + conv3d(conv3d(low1, dim2), dim2)

    low3 = conv3d(low2, dim)
    up2 = upnn3d(low3, x)

    out = up2 + x
    tf.add_to_collection("checkpoints", out)

    return out

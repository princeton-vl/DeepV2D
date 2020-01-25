import tensorflow as tf
import numpy as np
from utils.einsum import einsum


@tf.custom_gradient
def cholesky_solve(H, b):
    """Solves the linear system Hx = b"""
    chol = tf.linalg.cholesky(H)
    xx = tf.linalg.cholesky_solve(chol, b)

    # see OptNet: https://arxiv.org/pdf/1703.00443.pdf
    def grad(dx):
        # reuse choleksy factorization
        dz = tf.linalg.cholesky_solve(chol, dx)
        xs = tf.squeeze(xx,  -1)
        zs = tf.squeeze(dz, -1)
        dH = -einsum('...i,...j->...ij', xs, zs)
        return [dH, dz]

    return xx, grad

def solve(H, b, max_update=1.0):
    """ Solves the linear system Hx = b, H > 0"""

    # small system, solve on cpu
    with tf.device('/cpu:0'):
        H = tf.cast(H, tf.float64)
        b = tf.cast(b, tf.float64)

        b = tf.expand_dims(b, -1)
        x = cholesky_solve(H, b)

        # replaces nans and clip large updates
        bad_values = tf.is_nan(x)
        x = tf.where(bad_values, tf.zeros_like(x), x)
        x = tf.clip_by_value(x, -max_update, max_update)

        x = tf.squeeze(x, -1)
        x = tf.cast(x, tf.float32)
        
    return x


# def solve(H, b):
#     return tf.squeeze(tf.linalg.solve(H, tf.expand_dims(b, -1)), -1)
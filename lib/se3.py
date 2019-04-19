import tensorflow as tf
import numpy as np

from tensorflow.python.framework import function


def inv_SE3(G):
    """Inverts rigid body transformation"""
    batch, _, _ = G.get_shape().as_list()
    R = tf.transpose(G[:, 0:3, 0:3], [0, 2, 1])
    t = tf.expand_dims(G[:, 0:3, 3], -1)
    tp = -tf.matmul(R, t)

    filler = tf.constant([0.0, 0.0, 0.0, 1.0], shape=[1, 1, 4])
    filler = tf.tile(filler, [batch, 1, 1])
    Ginv = tf.concat([tf.concat([R, tp], axis=2), filler], axis=1)
    return Ginv


def solve_SE3(A, B):
    """Solves for the transformation which takes A->B, B=XA"""
    return tf.matmul(B, inv_SE3(A))


def skew_symmetric(x):
    """Converts vec to skew-symmetric matrix
    Args:
      x: vector -- size = [N, 3]
    Returns:
      Skew symmetric matrix -- size = [N, 3, 3]
    """
    a1, a2, a3 = x[:, 0], x[:, 1], x[:, 2]
    z = tf.zeros_like(a1)

    X = tf.stack([
        tf.stack([z, -a3, a2], axis=1),
        tf.stack([a3, z, -a1], axis=1),
        tf.stack([-a2, a1, z], axis=1),
    ], axis=1)

    return X


def exp_se3_jac(G):
    """Override exp mapping gradient
    Args:
      G: SE3 transform -- size = [N, 4, 4]
    Returns:
      Jacobian of G wrt \epsilon --size = [N, 3, 4, 6]
    """
    N = tf.shape(G)[0]
    R, t = G[:, 0:3, 0:3], G[:, 0:3, 3]

    filler_zero = tf.zeros((N, 3, 3))
    filler_diag = tf.eye(3, batch_shape=[N])

    J = tf.concat([
        tf.concat([filler_zero, skew_symmetric(-R[:, :, 0])], axis=2),
        tf.concat([filler_zero, skew_symmetric(-R[:, :, 1])], axis=2),
        tf.concat([filler_zero, skew_symmetric(-R[:, :, 2])], axis=2),
        tf.concat([filler_diag, skew_symmetric(-t)], axis=2)
    ], axis=1)


    return J


def se3_increment_grad(op, grad):
    """Approximates gradient of left update for small xi"""
    G, xi = op.inputs[0], op.inputs[1]
    N = tf.shape(G)[0]
    dG = exp_se3(xi)

    G_diff = tf.matmul(tf.transpose(dG, [0, 2, 1]), grad)

    jac = exp_se3_jac(G)
    top_diff = tf.reshape(tf.transpose(grad[:, 0:3], [0, 2, 1]), [N, 1, -1])
    xi_diff = tf.reshape(tf.matmul(top_diff, jac), [N, 6])

    return [G_diff, xi_diff]


def exp_se3(xi):
    """matrix exponial of se(3), uses taylor approximation when theta is small
    Inputs:
        xi: [batch, 6], element in se(3)
            (v, w), v: translation, w: rotation
    Output:
        G: [batch, 4, 4], expm(xi) element of SE(3) group
    """
    batch = tf.shape(xi)[0]
    v, w, = tf.split(xi, [3, 3], axis=1)

    # use double precision
    v = tf.cast(v, "float64")
    w = tf.cast(w, "float64")

    eps = 1e-4
    theta_sq = tf.reduce_sum(w**2, axis=1)
    theta_sq = tf.reshape(theta_sq, [-1, 1, 1])

    theta = tf.sqrt(theta_sq)
    theta_po4 = theta_sq * theta_sq

    wx = skew_symmetric(w)
    wx_sq = tf.matmul(wx, wx)
    I = tf.eye(3, batch_shape=[batch], dtype=tf.float64)

    R1 =  I + (1.0 - (1.0/6.0)*theta_sq + (1.0/120.0)*theta_po4)*wx + \
        (0.5 - (1.0/12.0)*theta_sq + (1.0/720.0)*theta_po4)*wx_sq

    R2 = I + (tf.sin(theta)/theta)*wx + ((1-tf.cos(theta))/theta_sq)*wx_sq

    V1 = I + (0.5 - (1.0/24.0)*theta_sq + (1.0/720.0)*theta_po4)*wx + \
        ((1.0/6.0) - (1.0/120.0)*theta_sq + (1.0/5040.0)*theta_po4)*wx_sq

    V2 = I + ((1-tf.cos(theta))/theta_sq)*wx + \
        ((theta-tf.sin(theta))/(theta_sq*theta))*wx_sq

    R = tf.where(theta[:, 0, 0]<eps, R1, R2)
    V = tf.where(theta[:, 0, 0]<eps, V1, V2)

    t = tf.matmul(V, tf.reshape(v, [-1, 3, 1]))
    fill = tf.constant([0, 0, 0, 1], dtype=tf.float64)

    fill = tf.reshape(fill, [1, 1, 4])
    fill = tf.tile(fill, [batch, 1, 1])

    G = tf.concat([R, t], axis=2)
    G = tf.concat([G, fill], axis=1)

    G = tf.cast(G, "float32")
    return G


@function.Defun(tf.float32, tf.float32,
        python_grad_func=se3_increment_grad,
        shape_func=lambda op: [op.inputs[0].shape])
def increment(G, xi):
    """ Left increment of rigid body transformation: G = expm(xi) G"""
    dG = exp_se3(xi)
    Gp = tf.matmul(dG, G)
    return Gp

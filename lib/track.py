import tensorflow as tf
import numpy as np

from tensorflow.contrib.distributions import percentile
from tensorflow.python.framework import function
import se3, camera
from core.config import cfg

MIN_DEPTH = 0.2
MAX_UPDATE = 0.1
MAX_GRADIENT = 1e-4

def clip_backprop_grad(op, grad):
    return [tf.where(tf.abs(grad)<MAX_GRADIENT, grad, tf.zeros_like(grad))]


@function.Defun(tf.float32,
                python_grad_func=clip_backprop_grad,
                shape_func=lambda op: [op.inputs[0].shape])
def clip_backprop(x):
    return x


def compute_jacobians(T, X, kv):

    X1 = tf.einsum('ajk,aik->aij', T, X)
    x1 = camera.proj(X1, kv)

    pX, pY, pZ, _ = tf.split(X1, [1, 1, 1, 1], axis=-1)
    fx, fy, cx, cy = tf.split(kv, [1, 1, 1, 1], axis=-1)

    fx = tf.reshape(fx, [-1, 1, 1])
    fy = tf.reshape(fy, [-1, 1, 1])
    cx = tf.reshape(cx, [-1, 1, 1])
    cy = tf.reshape(cy, [-1, 1, 1])

    zero = tf.zeros_like(pZ)
    ones = tf.ones_like(pZ)
    d = 1.0 / pZ

    J = tf.stack([
        tf.concat([fx*d, zero, -fx*pX*d**2, -fx*pX*pY*d**2, fx*(1+pX**2*d**2), -fx*pY*d], axis=-1),
        tf.concat([zero, fy*d, -fy*pY*d**2, -fy*(1+pY**2*d**2), fy*pX*pY*d**2,  fy*pX*d], axis=-1),
    ], axis=-2)

    return x1, J



def weighted_mean_residual(T, X, target, weight, kv, norm='l1'):
    X1 = tf.einsum('ajk,aik->aij', T, X)
    x1 = camera.proj(X1, kv)

    v = tf.to_float(
        (X1[...,2]>MIN_DEPTH) &
        ( X[...,2]>MIN_DEPTH)
    )

    if norm == 'l1':
        r = v[...,tf.newaxis] * weight * tf.abs(target-x1)

    elif norm == 'l2':
        r = v[...,tf.newaxis] * weight * (target-x1)**2

    return tf.reduce_mean(r, axis=[1,2])



def gn_step(T, X, target, weight, kv):

    batch, num, _ = X.get_shape().as_list()
    X1 = tf.einsum('ajk,aik->aij', T, X)
    x1, J = compute_jacobians(T, X, kv)

    resid = tf.subtract(target, x1)

    v = tf.to_float(
        (X1[...,2]>MIN_DEPTH) &
        ( X[...,2]>MIN_DEPTH)
    )

    R = tf.reshape(resid, [batch, 2*num, 1])
    W = tf.reshape(weight*v[...,tf.newaxis], [batch, 2*num, 1])

    J = tf.reshape(J, [batch, 2*num, 6])
    Jt = tf.transpose(J, [0, 2, 1])

    lm = cfg.MOTION.FLOWSE3.LM_LAMBDA
    ep = cfg.MOTION.FLOWSE3.EP_LAMBDA

    H = tf.matmul(Jt, W*J)
    b = tf.matmul(Jt, W*R)
    H += ep*tf.eye(6, batch_shape=[batch])

    update = tf.matrix_solve(H, b, adjoint=True)
    update = tf.reshape(update, [batch, 6])

    update = tf.clip_by_value(update, -MAX_UPDATE, MAX_UPDATE)
    T1 = se3.increment(T, update)

    return T1


def minimize_residual(G, X, target, weight, kv, n_steps=1):
    """
    Perform Gauss-Newton steps to minimize the geometric reprojection error

    Inputs:
        G: [batch, 4, 4], initial estimate of camera motion
        X: [batch, num, 3], point cloud
        target: [batch, num, 2], target projected coordinates
        weight: [batch, num, 2], weight for residual flow field
        kv: (fx,fy,cx,cy), camera intrinsics

    Outputs:
        G: updated camera motion estimate
        resid: residual reprojection error
    """


    weight = clip_backprop(weight)
    target = clip_backprop(target)

    # map weight to [0, 1]
    weight = tf.nn.sigmoid(weight)

    for i in range(n_steps):
        G1 = gn_step(G, X, target, weight, kv)
        resid0 = weighted_mean_residual( G, X, target, weight, kv)
        resid1 = weighted_mean_residual(G1, X, target, weight, kv)

        # # accept the update if it reduces the error
        accept_update = tf.to_float(resid1 < resid0)[:, tf.newaxis, tf.newaxis]
        G = accept_update*G1 + (1.0-accept_update)*G

    err = weighted_mean_residual(G, X, target, weight, kv, norm='l2')
    return G, err

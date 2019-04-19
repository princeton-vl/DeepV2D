import tensorflow as tf
import numpy as np


# def compute_errors(gt, pred, min_depth=0.1, max_depth=1e12, sc=False):
#     mask = np.logical_and(gt>min_depth, gt<max_depth)
#     gt = gt[mask]
#     pred = pred[mask]
#
#     if sc:
#         print (np.median(gt) / np.median(pred))
#         pred *= (np.median(gt) / np.median(pred))
#
#     thresh = np.maximum((gt / pred), (pred / gt))
#     a1 = (thresh < 1.25   ).mean()
#     a2 = (thresh < 1.25 ** 2).mean()
#     a3 = (thresh < 1.25 ** 3).mean()
#
#     rmse = (gt - pred) ** 2
#     rmse = np.sqrt(rmse.mean())
#
#     rmse_log = (np.log(gt) - np.log(pred)) ** 2
#     rmse_log = np.sqrt(rmse_log.mean())
#
#     abs_rel = np.mean(np.abs(gt - pred) / gt)
#     sq_rel = np.mean(((gt - pred)**2) / gt)
#
#     abs_rel = np.array(abs_rel, dtype="float32")
#     sq_rel = np.array(sq_rel, dtype="float32")
#     rmse = np.array(rmse, dtype="float32")
#     rmse_log = np.array(rmse_log, dtype="float32")
#     a1 = np.array(a1, dtype="float32")
#     a2 = np.array(a2, dtype="float32")
#     a3 = np.array(a3, dtype="float32")
#
#     l1_inv = np.mean(np.abs((1./gt - 1./pred)))
#
#     log_diff = np.log(gt) - np.log(pred)
#     num_pixels = gt.shape[0]
#     sc_inv = np.sqrt(np.sum(np.square(log_diff)) / num_pixels - np.square(np.sum(log_diff)) / np.square(num_pixels))
#     sc_inv = np.float32(sc_inv)
#
#     return abs_rel, l1_inv, sc_inv


def compute_errors(gt, pred, min_depth=0.1, max_depth=1e12, sc=False):
    mask = np.logical_and(gt>min_depth, gt<max_depth)
    gt = gt[mask]
    pred = pred[mask]

    if sc:
        print (np.median(gt) / np.median(pred))
        pred *= (np.median(gt) / np.median(pred))

    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25   ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred)**2) / gt)

    abs_rel = np.array(abs_rel, dtype="float32")
    sq_rel = np.array(sq_rel, dtype="float32")
    rmse = np.array(rmse, dtype="float32")
    rmse_log = np.array(rmse_log, dtype="float32")
    a1 = np.array(a1, dtype="float32")
    a2 = np.array(a2, dtype="float32")
    a3 = np.array(a3, dtype="float32")

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def eval(gt, pred):
    abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3 = \
        tf.py_func(compute_errors, [gt, pred], [tf.float32]*7)

    return {
        'abs_rel': abs_rel,
        'sq_rel': sq_rel,
        'rmse': rmse,
        'rmse_log': rmse_log,
        'a1': a1,
        'a2': a2,
        'a3': a3,
    }


# def eval(ystar, ypred, min_depth=0.1, max_depth=10.0):
#
#     ypred = tf.clip_by_value(ypred, min_depth, max_depth)
#
#     ystar_flatten = tf.reshape(ystar, [-1])
#     ypred_flatten = tf.reshape(ypred, [-1])
#
#     valid = tf.where((ystar_flatten>min_depth)&(ystar_flatten<max_depth))
#     ystar = tf.gather(ystar_flatten, valid)
#     ypred = tf.gather(ypred_flatten, valid)
#
#     ystar_log = tf.log(ystar)
#     ypred_log = tf.log(ypred)
#
#     alpha = tf.reduce_mean(ystar_log - ypred_log)
#     rmse_sc = tf.sqrt(tf.reduce_mean((ypred_log-ystar_log+alpha)**2))
#
#     # RMSE linear
#     rmse_lin = tf.sqrt(tf.reduce_mean((ypred-ystar)**2))
#     rmse_log = tf.sqrt(tf.reduce_mean((tf.log(ypred)-tf.log(ystar))**2))
#
#     delta = tf.maximum(ystar / ypred, ypred / ystar)
#     thrs_1 = tf.reduce_mean(tf.to_float(delta < 1.25**1))
#     thrs_2 = tf.reduce_mean(tf.to_float(delta < 1.25**2))
#     thrs_3 = tf.reduce_mean(tf.to_float(delta < 1.25**3))
#
#     metrics = {
#         'rmse_lin': rmse_lin,
#         'rmse_log': rmse_log,
#         'rmse_sc': rmse_sc,
#         'thrs_1': thrs_1,
#         'thrs_2': thrs_2,
#         'thrs_3': thrs_3,
#     }
#
#     return metrics

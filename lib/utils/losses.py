import tensorflow as tf
import se3
import camera

MAX_ERROR = 100
EPS = 0.01


def rotation_and_translation_errors(R1, t1, R2, t2):

    ri = tf.trace(tf.matmul(tf.transpose(R2, [0, 2, 1]), R1))
    angle = tf.acos(tf.minimum(1.0, tf.maximum(-1.0, (ri-1)/2)))
    rotation_error = tf.reduce_mean(angle)

    translation_error = tf.reduce_mean(tf.reduce_sum((t1-t2)**2, axis=-1))
    return rotation_error, translation_error


def motion_error(pose_star, pose_pred, depth, intrinsics, min_depth=0.75):

    R1, t1 = pose_star[:, 0:3, 0:3], pose_star[:, 0:3, 3]
    R2, t2 = pose_pred[:, 0:3, 0:3], pose_pred[:, 0:3, 3]

    rotation_error, translation_error = \
        rotation_and_translation_errors(R1, t1, R2, t2)

    depth = tf.squeeze(depth, axis=3)
    valid = tf.to_float(depth>min_depth)

    x1 = camera.camera_transform_project(pose_star, depth, intrinsics)
    x2 = camera.camera_transform_project(pose_pred, depth, intrinsics)

    geometric_diff = tf.reduce_sum((x1-x2)**2, axis=-1)
    val_mean = tf.reduce_mean(valid)

    geo_err = tf.losses.huber_loss(x1, x2, reduction=tf.losses.Reduction.NONE)
    geo_err = tf.clip_by_value(geo_err, -MAX_ERROR, MAX_ERROR)

    geometric_error = tf.reduce_mean(valid[..., tf.newaxis] * geo_err)
    geometric_error /= (val_mean + EPS)

    depth_flat = tf.reshape(depth, [-1])
    geom_flat = tf.reshape(geometric_diff, [-1])

    geom_valid = tf.gather(geom_flat, tf.where(depth_flat>min_depth))
    epe = tf.sqrt(geom_valid)

    metrics = {
        'translation_rmse': tf.sqrt(translation_error),
        'rotation_error': rotation_error,
        'mean_epe': tf.reduce_mean(epe),
        'median_epe': tf.contrib.distributions.percentile(epe, 50.0),
    }

    return geometric_error, metrics


def compute_weights_reg_loss(ws, k=2048):

    iters, batch, height, width, _ = ws.get_shape().as_list()
    ws = tf.transpose(ws, [0, 1, 4, 2, 3])
    ws = tf.reshape(ws, [iters*batch*2, height*width])
    top, _ = tf.nn.top_k(ws, k=k, sorted=False, name='topk')
    ref = tf.ones_like(top)

    l = tf.nn.sigmoid_cross_entropy_with_logits(labels=ref, logits=top)
    return tf.reduce_mean(l)

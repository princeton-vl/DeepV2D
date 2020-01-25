import tensorflow as tf
import numpy as np

def gather_nd(image, indicies, batch_dims=0):
    indicies_shape = tf.shape(indicies)
    batch_inds = [tf.range(indicies_shape[i]) for i in range(batch_dims)]
    batch_inds = tf.meshgrid(*batch_inds, indexing='ij')
    batch_inds = tf.stack(batch_inds, axis=-1)

    batch_shape, batch_tile = [], []
    for i in range(len(indicies.get_shape().as_list())-1):
        if i < batch_dims:
            batch_shape.append(indicies_shape[i])
            batch_tile.append(1)
        else:
            batch_shape.append(1)
            batch_tile.append(indicies_shape[i])

    batch_shape.append(batch_dims)
    batch_tile.append(1)
    batch_inds = tf.reshape(batch_inds, batch_shape)
    batch_inds = tf.tile(batch_inds, batch_tile)

    indicies = tf.concat([batch_inds, indicies], axis=-1)
    return tf.gather_nd(image, indicies)

def bilinear_sampler(image, coords, batch_dims=1, return_valid=False):
    """ performs bilinear sampling using coords grid """
    img_shape = tf.shape(image)
    coords_x, coords_y = tf.split(coords, [1, 1], axis=-1)

    x0 = tf.floor(coords_x)
    y0 = tf.floor(coords_y)
    x1 = x0 + 1.0
    y1 = y0 + 1.0

    w00 = (x1-coords_x) * (y1-coords_y)
    w01 = (coords_x-x0) * (y1-coords_y)
    w10 = (x1-coords_x) * (coords_y-y0)
    w11 = (coords_x-x0) * (coords_y-y0)

    x0 = tf.cast(x0, 'int32')
    x1 = tf.cast(x1, 'int32')
    y0 = tf.cast(y0, 'int32')
    y1 = tf.cast(y1, 'int32')

    x0c = tf.clip_by_value(x0, 0, img_shape[-2]-1)
    x1c = tf.clip_by_value(x1, 0, img_shape[-2]-1)
    y0c = tf.clip_by_value(y0, 0, img_shape[-3]-1)
    y1c = tf.clip_by_value(y1, 0, img_shape[-3]-1)

    valid = tf.equal(x0c, x0) & tf.equal(x1c, x1) & \
        tf.equal(y0c, y0) & tf.equal(y1c, y1)
    valid = tf.cast(valid, 'float32')

    coords00 = tf.concat([y0c,x0c], axis=-1)
    coords01 = tf.concat([y0c,x1c], axis=-1)
    coords10 = tf.concat([y1c,x0c], axis=-1)
    coords11 = tf.concat([y1c,x1c], axis=-1)

    img00 = gather_nd(image, coords00, batch_dims=batch_dims)
    img01 = gather_nd(image, coords01, batch_dims=batch_dims)
    img10 = gather_nd(image, coords10, batch_dims=batch_dims)
    img11 = gather_nd(image, coords11, batch_dims=batch_dims)

    out = w00*img00 + w01*img01 + w10*img10 + w11*img11
    if return_valid:
        return valid*out, valid
    
    return valid * out
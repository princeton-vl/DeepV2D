import tensorflow as tf



def bilinear_sampler_general(imgs, coords):
    """
    General case of bilinear sampling

    imgs: [B, H, W, _]
    coords: [N, H, W, 3]
    """

    batch, height, width, _ = imgs.get_shape().as_list()
    coords_i = coords[..., 0]
    coords_x = coords[..., 1]
    coords_y = coords[..., 2]

    coords_x = tf.where(tf.is_nan(coords_x), tf.zeros_like(coords_x), coords_x)
    coords_y = tf.where(tf.is_nan(coords_y), tf.zeros_like(coords_y), coords_y)

    v = tf.to_float(
        (coords_x > 0) &
        (coords_x < width-1) &
        (coords_y > 0) &
        (coords_y < height-1)
    )
    v = tf.expand_dims(v, axis=-1)

    coords_x = tf.clip_by_value(coords_x, 0, width-1)
    coords_y = tf.clip_by_value(coords_y, 0, height-1)


    x0 = tf.floor(coords_x)
    x1 = x0 + 1
    y0 = tf.floor(coords_y)
    y1 = y0 + 1

    x0 = tf.cast(x0, 'int32')
    x1 = tf.cast(x1, 'int32')
    y0 = tf.cast(y0, 'int32')
    y1 = tf.cast(y1, 'int32')
    coords_i = tf.cast(coords_i, 'int32')

    coords00 = tf.stack([coords_i, y0, x0], axis=-1)
    coords01 = tf.stack([coords_i, y0, x1], axis=-1)
    coords10 = tf.stack([coords_i, y1, x0], axis=-1)
    coords11 = tf.stack([coords_i, y1, x1], axis=-1)

    img00 = tf.gather_nd(imgs, coords00)
    img01 = tf.gather_nd(imgs, coords01)
    img10 = tf.gather_nd(imgs, coords10)
    img11 = tf.gather_nd(imgs, coords11)

    dx = coords_x - tf.cast(x0, 'float32')
    dy = coords_y - tf.cast(y0, 'float32')
    dx = tf.expand_dims(dx, axis=-1)
    dy = tf.expand_dims(dy, axis=-1)

    w00 = (1.0 - dy) * (1.0 - dx)
    w01 = (1.0 - dy) * dx
    w10 = dy * (1.0 - dx)
    w11 = dy * dx

    output = v * tf.add_n([
        w00 * img00,
        w01 * img01,
        w10 * img10,
        w11 * img11,
    ])

    return output



def bilinear_sampler(imgs, coords):
    """
    imgs: [B, H, W, D]
    coords: [B, H, W, 2]
    """

    batch, height, width, _ = imgs.get_shape().as_list()

    coords_i = tf.reshape(tf.range(batch), [batch, 1, 1, 1])
    coords_i = tf.tile(coords_i, [1, height, width, 1])
    coords_i = tf.cast(coords_i, 'float32')

    coords = tf.concat([coords_i, coords], axis=-1)
    output = bilinear_sampler_general(imgs, coords)

    return output


def bilinear_sampler_nd(imgs, coords):
    """
    imgs: [B, H, W, D]
    coords: [B, H, W, S, 2]
    """

    batch, height, width, samples, _ = coords.get_shape().as_list()
    coords_i = tf.reshape(tf.range(batch), [batch, 1, 1, 1, 1])
    coords_i = tf.tile(coords_i, [1, height, width, samples, 1])
    coords_i = tf.cast(coords_i, 'float32')

    coords = tf.concat([coords_i, coords], axis=-1)
    output = bilinear_sampler_general(imgs, coords)

    return output

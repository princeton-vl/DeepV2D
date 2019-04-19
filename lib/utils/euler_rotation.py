import tensorflow as tf
import numpy as np

def euler2mat(z, y, x):
      """Converts euler angles to rotation matrix
       TODO: remove the dimension for 'N' (deprecated for converting all source
             poses altogether)
       Reference: https://github.com/pulkitag/pycaffe-utils/blob/master/rot_utils.py#L174
      Args:
          z: rotation angle along z axis (in radians) -- size = [B, N]
          y: rotation angle along y axis (in radians) -- size = [B, N]
          x: rotation angle along x axis (in radians) -- size = [B, N]
      Returns:
          Rotation matrix corresponding to the euler angles -- size = [B, N, 3, 3]
      """
      B = tf.shape(z)[0]
      N = 1
      z = tf.clip_by_value(z, -np.pi, np.pi)
      y = tf.clip_by_value(y, -np.pi, np.pi)
      x = tf.clip_by_value(x, -np.pi, np.pi)

      # Expand to B x N x 1 x 1
      z = tf.expand_dims(tf.expand_dims(z, -1), -1)
      y = tf.expand_dims(tf.expand_dims(y, -1), -1)
      x = tf.expand_dims(tf.expand_dims(x, -1), -1)

      zeros = tf.zeros([B, N, 1, 1])
      ones  = tf.ones([B, N, 1, 1])

      cosz = tf.cos(z)
      sinz = tf.sin(z)
      rotz_1 = tf.concat([cosz, -sinz, zeros], axis=3)
      rotz_2 = tf.concat([sinz,  cosz, zeros], axis=3)
      rotz_3 = tf.concat([zeros, zeros, ones], axis=3)
      zmat = tf.concat([rotz_1, rotz_2, rotz_3], axis=2)

      cosy = tf.cos(y)
      siny = tf.sin(y)
      roty_1 = tf.concat([cosy, zeros, siny], axis=3)
      roty_2 = tf.concat([zeros, ones, zeros], axis=3)
      roty_3 = tf.concat([-siny,zeros, cosy], axis=3)
      ymat = tf.concat([roty_1, roty_2, roty_3], axis=2)

      cosx = tf.cos(x)
      sinx = tf.sin(x)
      rotx_1 = tf.concat([ones, zeros, zeros], axis=3)
      rotx_2 = tf.concat([zeros, cosx, -sinx], axis=3)
      rotx_3 = tf.concat([zeros, sinx, cosx], axis=3)
      xmat = tf.concat([rotx_1, rotx_2, rotx_3], axis=2)

      rotMat = tf.matmul(tf.matmul(xmat, ymat), zmat)
      return rotMat


def pose_vec2mat(vec, use_filler=True):
      """Converts 6DoF parameters to transformation matrix
      Args:
          vec: 6DoF parameters in the order of tx, ty, tz, rx, ry, rz -- [B, 6]
      Returns:
          A transformation matrix -- [B, 4, 4]
      """
      batch_size, _ = vec.get_shape().as_list()
      translation = tf.slice(vec, [0, 0], [-1, 3])
      translation = tf.expand_dims(translation, -1)
      rx = tf.slice(vec, [0, 3], [-1, 1])
      ry = tf.slice(vec, [0, 4], [-1, 1])
      rz = tf.slice(vec, [0, 5], [-1, 1])
      rot_mat = euler2mat(rz, ry, rx)
      rot_mat = tf.squeeze(rot_mat, axis=[1])
      filler = tf.constant([0.0, 0.0, 0.0, 1.0], shape=[1, 1, 4])
      filler = tf.tile(filler, [batch_size, 1, 1])
      transform_mat = tf.concat([rot_mat, translation], axis=2)
      if use_filler:
          transform_mat = tf.concat([transform_mat, filler], axis=1)
      return transform_mat

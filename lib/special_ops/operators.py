import tensorflow as tf
import os.path as osp

from tensorflow.python.framework import ops
from utils import bilinear_sampler



try:
    filename = osp.join(osp.dirname(__file__), 'backproject.so')
    _backproject_module = tf.load_op_library(filename)

    back_project = _backproject_module.back_project
    back_project_grad = _backproject_module.back_project_grad

    @ops.RegisterGradient("BackProject")
    def _back_project_grad(op, grad):
        inputs = op.inputs[0]
        coords = op.inputs[1]
        inputs_grad, coords_grad = back_project_grad(inputs, coords, grad)

        return [inputs_grad, coords_grad]

    use_python = False

except:
    print("Backprojection Op not available: Using python implementation")
    use_python = True


def backproject_py(inputs, coords):
    batch, frames, height, width, depth, _ = coords.get_shape().as_list()

    vols = []
    for i in range(frames):
        vol = bilinear_sampler.bilinear_sampler_nd(inputs[:, i], coords[:, i])
        vols.append(vol)

    matching_vol = tf.concat(vols, axis=-1)
    return matching_vol


def backproject(inputs, coords):
    with tf.name_scope("backproject"):
        if use_python:
            matching_vol = backproject_py(inputs, coords)

        else:
            b, f, h, w, d, _ = coords.get_shape().as_list()
            c = inputs.get_shape().as_list()[-1]

            inputs = tf.transpose(inputs, [0, 2, 3, 1, 4])
            coords = tf.transpose(coords, [0, 2, 3, 4, 1, 5])

            matching_vol = back_project(inputs, coords)
            matching_vol = tf.reshape(matching_vol, [b, h, w, d, f*c])

    return matching_vol

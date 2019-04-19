import tensorflow as tf
import numpy as np

from networks.structure import StructureNetwork
from networks.motion import MotionNetwork
from fcrn import fcrn


class DeepV2D(object):

    def __init__(self, dims, cfg, use_fcrn=False):
        self.dims = dims
        frames, height, width = self.dims

        images = tf.placeholder(tf.float32, [1, frames, height, width, 3])
        depth = tf.placeholder(tf.float32, [1, height, width, 1])
        poses = tf.placeholder(tf.float32, [1, frames, 4, 4])
        intrinsics = tf.placeholder(tf.float32, [1, 4])
        do_init = tf.placeholder(tf.bool, shape=())

        self.placeholders = {
            'images': images,
            'poses': poses,
            'depth': depth,
            'intrinsics': intrinsics,
            'do_init': do_init,
        }

        if use_fcrn:
            self.single_depth = self.build_fcrn_module(images[:, 0])

        # DeepV2D iteration
        motion = MotionNetwork(cfg.MOTION, is_training=False)
        stereo = StructureNetwork(cfg.STRUCTURE, is_training=False)

        # get the camera motion between the keyframe and each of the other images
        image_star = images[:, 0]
        poses_pred = motion.forward(images[:, 1:], image_star, depth, intrinsics)

        """
        During initialization, we predict depth using only the keyframe
        """
        images_init = tf.tile(image_star[:, tf.newaxis], [1,frames,1,1,1])
        poses_init = tf.eye(4, batch_shape=[1, frames])

        # if this is the first iteration, use pose regression network
        images = tf.cond(do_init, lambda: images_init, lambda: images)
        dP = tf.cond(do_init, lambda: poses_init, lambda: poses_pred)

        # plug the motions into the stereo network
        stereo_depth = stereo.forward(images, dP, intrinsics)

        self.outputs = {
            'depth': tf.expand_dims(stereo_depth, -1),
            'motion': poses_pred,
        }


    def build_fcrn_module(self, image):
        batch, heigth, width, _ = image.get_shape().as_list()
        with tf.variable_scope("FCRN") as scope:
            # crop out boarder and flip color channels
            fcrn_input = tf.image.resize_area(image[:, 4:-4, 6:-6, ::-1], [228, 304])
            net = fcrn.ResNet50UpProj({'data': fcrn_input}, batch, 1, False)
            dpred = tf.stop_gradient(net.get_output())
            dpred = tf.image.resize_bilinear(dpred, [heigth, width])

        return dpred


    def set_fcrn_weights(self, sess):
        fcrn_vars = {}
        for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="FCRN"):
            fcrn_vars[var.name.replace('FCRN/', '').replace(':0', '')] = var

        fcrn_saver = tf.train.Saver(fcrn_vars)
        fcrn_saver.restore(sess, 'models/NYU_FCRN.ckpt')


    def restore(self, sess, ckpt):
        init_op = tf.group(tf.global_variables_initializer(),
                   tf.local_variables_initializer())

        self.sess = sess
        self.sess.run(init_op)

        saver =  tf.train.Saver([var for var in tf.model_variables()])
        saver.restore(sess, ckpt)


    def forward(self, data_blob, iters=5, init_mode='constant'):

        CONSTANT = 4.0
        if init_mode == 'constant':
            depth = CONSTANT*np.ones([1, self.dims[1], self.dims[2], 1])

        elif init_mode == 'fcrn':
            feed_dict = {
                self.placeholders['images']: data_blob['images'][np.newaxis]
            }
            depth = self.sess.run(self.single_depth, feed_dict=feed_dict)

        predictions = [depth]
        for i in range(iters):
            do_init = (not init_mode=='fcrn') and (i==0)
            feed_dict = {
                self.placeholders['images']: data_blob['images'][np.newaxis],
                self.placeholders['intrinsics']: data_blob['intrinsics'][np.newaxis],
                self.placeholders['depth']: depth,
                self.placeholders['do_init']: do_init,
            }

            output = self.sess.run(self.outputs, feed_dict=feed_dict)
            depth = output['depth']
            pose = output['motion']
            predictions.append(depth)

        return predictions

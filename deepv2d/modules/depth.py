import tensorflow as tf
import numpy as np
slim = tf.contrib.slim

from .networks import hg
from .networks.layer_ops import *

from geometry.transformation import *
from geometry.intrinsics import *
from special_ops import operators

def add_depth_summaries(gt, pr):
    gt = tf.reshape(gt, [-1])
    pr = tf.reshape(pr, [-1])

    v = tf.where((gt>0.1) & (gt<10.0))
    gt = tf.gather(gt, v)
    pr = tf.gather(pr, v)

    thresh = tf.maximum(gt / pr, pr / gt)
    delta = tf.reduce_mean(tf.to_float(thresh < 1.25))
    abs_rel = tf.reduce_mean(tf.abs(gt-pr) / gt)

    with tf.device('/cpu:0'):
        tf.summary.scalar("a1", delta)
        tf.summary.scalar("rel", abs_rel)


class DepthNetwork(object):
    def __init__(self, cfg, schedule=None, is_training=True, reuse=False):
        self.cfg = cfg
        self.reuse = reuse
        self.is_training = is_training
        self.schedule = schedule

        self.summaries = {}
        self.depths = tf.lin_space(cfg.MIN_DEPTH, cfg.MAX_DEPTH, cfg.COST_VOLUME_DEPTH)

        self.batch_norm_params = {
          'decay': .995,
          'epsilon': 1e-5,
          'scale': True,
          'renorm': True,
          'renorm_clipping': schedule,
          'trainable': self.is_training,
          'is_training': self.is_training,
        }


    def encoder(self, inputs, reuse=False):
        """ 2D feature extractor """

        batch, frames, ht, wd, _ = tf.unstack(tf.shape(inputs), num=5)
        inputs = tf.reshape(inputs, [batch*frames, ht, wd, 3])

        with tf.variable_scope("encoder") as sc:
            with slim.arg_scope([slim.batch_norm], **self.batch_norm_params):
                with slim.arg_scope([slim.conv2d],
                                    weights_regularizer=slim.l2_regularizer(0.00005),
                                    normalizer_fn=None,
                                    activation_fn=None,
                                    reuse=reuse):

                    net = slim.conv2d(inputs, 32, [7, 7], stride=2)

                    net = res_conv2d(net, 32, 1)
                    net = res_conv2d(net, 32, 1)
                    net = res_conv2d(net, 32, 1)
                    net = res_conv2d(net, 64, 2)
                    net = res_conv2d(net, 64, 1)
                    net = res_conv2d(net, 64, 1)
                    net = res_conv2d(net, 64, 1)

                    net = hg.hourglass_2d(net, 4, 64)
                    net = hg.hourglass_2d(net, 4, 64)

                    embd = slim.conv2d(net, 32, [1, 1])

        embd = tf.reshape(embd, [batch, frames, ht//4, wd//4, 32])
        return embd

    def stereo_head(self, x):
        """ Predict probability volume from hg features"""
        x = bnrelu(x)
        x = slim.conv3d(x, 32, [3, 3, 3], activation_fn=tf.nn.relu)
        x = slim.conv3d(x, 32, [3, 3, 3], activation_fn=tf.nn.relu)
        tf.add_to_collection("checkpoints", x)

        logits = slim.conv3d(x, 1, [1, 1, 1], activation_fn=None)
        logits = tf.squeeze(logits, axis=-1)

        logits = tf.image.resize_bilinear(logits, self.input_dims)
        return logits

    def soft_argmax(self, prob_volume):
        """ Convert probability volume into point estimate of depth"""
        prob_volume = tf.nn.softmax(prob_volume, axis=-1)
        pred = tf.reduce_sum(self.depths*prob_volume, axis= -1)
        return pred

    def stereo_network_avg(self, Ts, images, intrinsics, adj_list=None):
        """3D Matching Network with view pooling
        Ts: collection of pose estimates correponding to images
        images: rgb images
        intrinsics: image intrinsics
        adj_list: [n, m] matrix specifying frames co-visiblee frames
        """

        cfg = self.cfg
        depths = tf.lin_space(cfg.MIN_DEPTH, cfg.MAX_DEPTH, cfg.COST_VOLUME_DEPTH)
        intrinsics = intrinsics_vec_to_matrix(intrinsics / 4.0)

        with tf.variable_scope("stereo", reuse=self.reuse) as sc:
            # extract 2d feature maps from images and build cost volume
            fmaps = self.encoder(images)
            volume = operators.backproject_avg(Ts, depths, intrinsics, fmaps, adj_list)

            self.spreds = []
            with slim.arg_scope([slim.batch_norm], **self.batch_norm_params):
                with slim.arg_scope([slim.conv3d],
                                    weights_regularizer=slim.l2_regularizer(0.00005),
                                    normalizer_fn=None,
                                    activation_fn=None):

                    dim = tf.shape(volume)
                    volume = tf.reshape(volume, [dim[0]*dim[1], dim[2], dim[3], dim[4], 64])

                    x = slim.conv3d(volume, 32, [1, 1, 1])
                    tf.add_to_collection("checkpoints", x)

                    # multi-view convolution
                    x = tf.add(x, conv3d(conv3d(x, 32), 32))

                    x = tf.reshape(x, [dim[0], dim[1], dim[2], dim[3], dim[4], 32])
                    x = tf.reduce_mean(x, axis=1)
                    tf.add_to_collection("checkpoints", x)

                    self.pred_logits = []
                    for i in range(self.cfg.HG_COUNT):
                        with tf.variable_scope("hg1_%d"%i):
                            x = hg.hourglass_3d(x, 4, 32)
                            self.pred_logits.append(self.stereo_head(x))

        return self.soft_argmax(self.pred_logits[-1])

    def stereo_network_cat(self, Ts, images, intrinsics):
        """3D Matching Network with view concatenation"""

        cfg = self.cfg
        depths = tf.lin_space(cfg.MIN_DEPTH, cfg.MAX_DEPTH, cfg.COST_VOLUME_DEPTH)
        intrinsics = intrinsics_vec_to_matrix(intrinsics / 4.0)

        with tf.variable_scope("stereo", reuse=self.reuse) as sc:
            # extract 2d feature maps from images and build cost volume
            fmaps = self.encoder(images)
            volume = operators.backproject_cat(Ts, depths, intrinsics, fmaps)

            self.spreds = []
            with slim.arg_scope([slim.batch_norm], **self.batch_norm_params):
                with slim.arg_scope([slim.conv3d],
                                    weights_regularizer=slim.l2_regularizer(0.00005),
                                    normalizer_fn=None,
                                    activation_fn=None):


                    x = slim.conv3d(volume, 48, [3, 3, 3])
                    x = tf.add(x, conv3d(conv3d(x, 48), 48))

                    self.pred_logits = []
                    for i in range(self.cfg.HG_COUNT):
                        with tf.variable_scope("hg1_%d"%i):
                            x = hg.hourglass_3d(x, 4, 48)
                            self.pred_logits.append(self.stereo_head(x))

        return self.soft_argmax(self.pred_logits[-1])


    def forward(self, poses, images, intrinsics, idx=None):

        images = 2 * (images / 255.0) - 1.0
        ht = images.get_shape().as_list()[2]
        wd = images.get_shape().as_list()[3]
        self.input_dims = [ht, wd]

        # perform per-view average pooling
        if self.cfg.MODE == 'avg':
            spred = self.stereo_network_avg(poses, images, intrinsics, idx)

        # perform view concatenation
        elif self.cfg.MODE == 'concat':
            spred = self.stereo_network_cat(poses, images, intrinsics)

        return spred


    def compute_loss(self, depth_gt, log_error=True):

        b_gt, h_gt, w_gt, _ = depth_gt.get_shape().as_list()

        total_loss = 0.0
        for i, logits in enumerate(self.pred_logits):

            pred = self.soft_argmax(logits)
            pred = tf.image.resize_bilinear(pred[...,tf.newaxis], [h_gt, w_gt])

            pred = tf.squeeze(pred, axis=-1)
            gt = tf.squeeze(depth_gt, axis=-1)

            valid = tf.to_float(gt>0.0)
            s = 1.0 / (tf.reduce_mean(valid) + 1e-8)

            gx = pred[:, :, 1:] - pred[:, :, :-1]
            gy = pred[:, 1:, :] - pred[:, :-1, :]
            vx = valid[:, :, 1:] * valid[:, :, :-1]
            vy = valid[:, 1:, :] * valid[:, :-1, :]

            # take l1 smoothness loss where gt depth is missing
            loss_smooth = \
                tf.reduce_mean((1-vx)*tf.abs(gx)) + \
                tf.reduce_mean((1-vy)*tf.abs(gy))

            loss_depth = s*tf.reduce_mean(valid*tf.abs(gt-pred))
            loss_i = self.cfg.TRAIN.SMOOTH_W * loss_smooth + loss_depth

            w = .5**(len(self.pred_logits)-i-1)
            total_loss += w * loss_i

        if log_error:
            add_depth_summaries(gt, pred)
        
        return total_loss

import tensorflow as tf
import numpy as np
slim = tf.contrib.slim

import camera
from special_ops import operators
from networks.layer_ops import *
from utils import bilinear_sampler



class StructureNetwork(object):
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
          'is_training': self.is_training,
        }


    def _hourglass_2d(self, x, n, dim, expand=64):
        dim2 = dim + expand
        x = x + conv2d(conv2d(x, dim), dim)

        pool1 = slim.max_pool2d(x, [2, 2], padding='SAME')

        low1 = conv2d(pool1, dim2)
        if n>1:
            low2 = self._hourglass_2d(low1, n-1, dim2)
        else:
            low2 = conv2d(low1, dim2)

        low3 = conv2d(low2, dim)
        up2 = upnn2d(low3, x)

        out = up2 + x
        return out

    def _hourglass_3d(self, x, n, dim, expand=48):
        dim2 = dim + expand

        x = x + conv3d(conv3d(x, dim), dim)
        pool1 = slim.max_pool3d(x, [2, 2, 2], padding='SAME')

        low1 = conv3d(pool1, dim2)
        if n>1:
            low2 = self._hourglass_3d(low1, n-1, dim2)
        else:
            low2 = low1 + conv3d(conv3d(low1, dim2), dim2)

        low3 = conv3d(low2, dim)
        up2 = upnn3d(low3, x)

        out = up2 + x
        return out


    def encoder(self, inputs, reuse=False):
        """ 2D feature extractor """

        batch, frames, height, width, _ = inputs.get_shape().as_list()
        inputs = tf.reshape(inputs, [batch*frames, height, width, -1])

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

                    net = self._hourglass_2d(net, 4, 64)
                    net = self._hourglass_2d(net, 4, 64)

                    embd = slim.conv2d(net, 32, [1, 1])

        embd = tf.reshape(embd, [batch, frames, int(height/4), int(width/4), -1])
        return embd


    def build_cvolume(self, Xs, Gs, intrinsics):

        batch, frames, height, width, _ = Xs.get_shape().as_list()
        depths = tf.reshape(self.depths, [1, 1, 1, -1])
        depths = tf.tile(depths, [batch, height, width, 1])

        # backproject point cloud
        x0 = camera.coords_grid(batch, height, width)
        X0 = camera.iproj(tf.expand_dims(x0, -2), depths, intrinsics)

        # transform/project the point cloud
        """
        a: batch_dim
        i: frame_dim
        p: depth_dim
        q: height_dim
        r: width_dim
        -> [a, i, p, q, r]
        """
        X1 = tf.einsum('aijk,apqrk->aipqrj', Gs, X0)
        x1 = camera.proj(X1, intrinsics)

        volume = operators.backproject(Xs, x1)
        return volume

    def stereo_head(self, x):
        """ Predict probability volume from hg features"""
        x = bnrelu(x)
        x = slim.conv3d(x, 32, [3, 3, 3], activation_fn=tf.nn.relu)
        x = slim.conv3d(x, 32, [3, 3, 3], activation_fn=tf.nn.relu)

        logits = slim.conv3d(x, 1, [1, 1, 1], activation_fn=None)
        logits = tf.squeeze(logits, axis=-1)

        logits = tf.image.resize_bilinear(logits, self.input_dims)
        return logits


    def soft_argmax(self, prob_volume):
        """ Convert probability volume into point estimate of depth"""
        prob_volume = tf.nn.softmax(prob_volume, axis=-1)
        pred = tf.reduce_sum(self.depths*prob_volume, axis=-1)
        return pred


    def stereo_network_avg(self, images, poses, intrinsics, idx=0):
        """3D Matching Network with view pooling"""

        batch, frames, height, width, _ = images.get_shape().as_list()
        with tf.variable_scope("stereo", reuse=self.reuse) as sc:
            feats = self.encoder(images)

            volumes = []
            for i in range(1, frames):
                dPs = tf.gather(poses, [idx, i], axis=1)
                Xs = tf.gather(feats, [idx, i], axis=1)
                vol = self.build_cvolume(Xs, dPs, intrinsics/4.0)
                volumes.append(vol)

            self.spreds = []
            with slim.arg_scope([slim.batch_norm], **self.batch_norm_params):
                with slim.arg_scope([slim.conv3d],
                                    weights_regularizer=slim.l2_regularizer(0.00005),
                                    normalizer_fn=None,
                                    activation_fn=None):

                    volume = tf.stack(volumes, axis=1)
                    dim = volume.get_shape().as_list()

                    volume = tf.reshape(volume, [dim[0]*dim[1], dim[2], dim[3], dim[4], -1])

                    x = slim.conv3d(volume, 32, [1, 1, 1])
                    x = tf.add(x, conv3d(conv3d(x, 32), 32))

                    x = tf.reshape(x, [dim[0], dim[1], dim[2], dim[3], dim[4], -1])
                    x = tf.reduce_mean(x, axis=1)

                    self.pred_logits = []
                    for i in range(self.cfg.HG_COUNT):
                        with tf.variable_scope("hg1_%d"%i):
                            x = self._hourglass_3d(x, 4, 32)
                            self.pred_logits.append(self.stereo_head(x))

        return self.soft_argmax(self.pred_logits[-1])


    def stereo_network_cat(self, images, poses, intrinsics):
        """3D Matching Network with view concatenation"""

        batch, frames, height, width, _ = images.get_shape().as_list()
        with tf.variable_scope("stereo", reuse=self.reuse) as sc:

            feats = self.encoder(images)
            volume = self.build_cvolume(feats, poses, intrinsics/4.0)

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
                            x = self._hourglass_3d(x, 4, 48)
                            self.pred_logits.append(self.stereo_head(x))

        return self.soft_argmax(self.pred_logits[-1])



    def forward(self, images, poses, intrinsics):

        images = 2*(tf.to_float(images)/255.0 - 0.5)
        batch, frames, height, width, _ = images.get_shape().as_list()

        self.input_dims = [height, width]

        # perform per-view average pooling
        if self.cfg.MODE == 'avg':
            spred = self.stereo_network_avg(images, poses, intrinsics)

        # perform view concatenation
        elif self.cfg.MODE == 'concat':
            spred = self.stereo_network_cat(images, poses, intrinsics)

        return spred


    def compute_loss(self, depth_gt):

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

        return total_loss

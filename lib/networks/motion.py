import tensorflow as tf
import numpy as np
slim = tf.contrib.slim

from networks.pose_regressors import pose_regressor_factory
from networks.layer_ops import *
from utils import losses
from utils import bilinear_sampler

import se3
import camera
import track


def pose_perturbation(G, delta):
    xi_dim = G.get_shape().as_list()[:-2] + [6,]
    dxi = tf.random_normal(xi_dim)
    G1 = se3.increment(G, dxi*delta)
    return G1


class MotionNetwork(object):

    def __init__(self, cfg, schedule=None, is_training=True, reuse=False, scope="Motion"):
        self.cfg = cfg
        self.reuse = reuse
        self.scope = scope
        self.is_training = is_training

        self.summaries = {}

        self.flows = []
        self.weights = []
        self.resids = []

        if schedule is None:
            self.batch_norm_params = {
                'is_training': self.is_training,
            }

        else:
            self.batch_norm_params = {
              'decay': .995,
              'epsilon': 1e-5,
              'scale': True,
              'renorm': True,
              'renorm_clipping': schedule,
              'is_training': self.is_training,
            }

    def forward(self, images, image_star, depth_star, intrinsics, G0=None):

        cfg = self.cfg
        batch, frames, height, width, _ = images.get_shape().as_list()

        SC = 4 # features are at 1/4 resolution
        with tf.name_scope("prepare_inputs"):
            depth1 = tf.tile(depth_star[:,tf.newaxis], [1,frames,1,1,1])
            depth1 = tf.reshape(depth1, [batch*frames, height, width])

            intrinsics = tf.tile(intrinsics[:, tf.newaxis], [1, frames, 1])
            intrinsics = tf.reshape(intrinsics, [batch*frames, 4])


        Gs = []
        with tf.variable_scope("motion", reuse=self.reuse):
            if G0 is None:
                G = pose_regressor_factory(image_star, images, intrinsics, cfg)
                Gs.append(G)
            else:
                G = tf.reshape(G0, [batch*frames, 4, 4])

            # random perturbation of camera pose
            if self.is_training:
                G = pose_perturbation(G, cfg.TRAIN.DELTA)

            # extract featurs and flatten the frame dimension
            feats, feat_star = self.encoder(images, image_star)
            feat1 = tf.tile(feat_star[:,tf.newaxis], [1,frames,1,1,1])
            feat1 = tf.reshape(feat1, [batch*frames, int(height/SC), int(width/SC), -1])
            feat2 = tf.reshape(feats, [batch*frames, int(height/SC), int(width/SC), -1])

            for i in range(cfg.FLOWSE3.ITER_COUNT):
                G = self.flowse3(feat1, feat2, depth1, intrinsics/SC, G=G, reuse=i>0)
                Gs.append(G)

            for i in range(len(Gs)):
                Gs[i] = tf.reshape(Gs[i], [batch, frames, 4, 4])
                dI = tf.eye(4, batch_shape=[batch, 1])
                Gs[i] = tf.concat([dI, Gs[i]], axis=1)


        self.resids = tf.stack(self.resids, axis=0)
        self.weights = tf.stack(self.weights, axis=0)
        self.flow = tf.stack(self.flows, axis=0)

        self.dGs = Gs
        return Gs[-1]


    def flowse3(self, feat1, feat2, depth, intrinsics, G, reuse=False):

        G = tf.stop_gradient(G)
        batch, height, width, _ = feat1.get_shape().as_list()
        depth  = resize_depth(depth, [height, width], min_depth=1e-3)

        coords = camera.camera_transform_project(G, depth, intrinsics)
        featw = bilinear_sampler.bilinear_sampler(feat2, coords)

        with tf.name_scope("residual"):
            inputs = tf.concat([feat1, featw], axis=-1)
            flow, weight = self.flownet(inputs, reuse=reuse)
            target = tf.add(coords, flow)

        with tf.name_scope("PnP"):
            X = camera.point_cloud_from_depth(depth, intrinsics)
            X_flat = tf.reshape(X, [batch, height*width, 4])
            target_flat = tf.reshape(target, [batch, height*width, 2])
            weight_flat = tf.reshape(weight, [batch, height*width, 2])

            G, resid = track.minimize_residual(
                G, X_flat, target_flat, weight_flat, intrinsics)

        # append intermediate outputs
        self.flows.append(flow)
        self.weights.append(weight)
        self.resids.append(resid)

        return G


    def encoder(self, images, image_star, reuse=False):

        batch, frames, height, width, _ = images.get_shape().as_list()
        images_stack = tf.concat([image_star[:, tf.newaxis], images], axis=1)
        inputs = tf.reshape(images_stack, [batch*(frames+1), height, width, -1])

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
                    net = slim.conv2d(net, 64, [1, 1], stride=1)

        feat_stack = tf.reshape(net, [batch, frames+1, int(height/4), int(width/4), -1])
        feat_star, feats = feat_stack[:, 0], feat_stack[:, 1:]

        return feats, feat_star


    def flownet(self, inputs, reuse=False):
        with tf.variable_scope('flow_net') as sc:
            end_points_collection = sc.original_name_scope + '_end_points'
            with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                                normalizer_fn=None,
                                reuse=reuse,
                                weights_regularizer=slim.l2_regularizer(0.0005),
                                activation_fn=tf.nn.relu,
                                outputs_collections=end_points_collection):

                cnv1  = slim.conv2d(inputs, 32,  [7, 7], stride=2, scope='cnv1')
                cnv1b = slim.conv2d(cnv1,  32,  [7, 7], stride=1, scope='cnv1b')
                cnv2  = slim.conv2d(cnv1b, 64,  [5, 5], stride=2, scope='cnv2')
                cnv2b = slim.conv2d(cnv2,  64,  [5, 5], stride=1, scope='cnv2b')
                cnv3  = slim.conv2d(cnv2b, 128, [3, 3], stride=2, scope='cnv3')
                cnv3b = slim.conv2d(cnv3,  128, [3, 3], stride=1, scope='cnv3b')
                cnv4  = slim.conv2d(cnv3b, 256, [3, 3], stride=2, scope='cnv4')
                cnv4b = slim.conv2d(cnv4,  256, [3, 3], stride=1, scope='cnv4b')
                cnv5  = slim.conv2d(cnv4b, 512, [3, 3], stride=2, scope='cnv5')
                cnv5b = slim.conv2d(cnv5,  512, [3, 3], stride=1, scope='cnv5b')

                upcnv5 = slim.conv2d_transpose(cnv5b, 256, [3, 3], stride=2, scope='upcnv5')
                upcnv5 = resize_like(upcnv5, cnv4b)
                i5_in  = tf.concat([upcnv5, cnv4b], axis=3)
                icnv5  = slim.conv2d(i5_in, 256, [3, 3], stride=1, scope='icnv5')

                upcnv4 = slim.conv2d_transpose(icnv5, 128, [3, 3], stride=2, scope='upcnv4')
                upcnv4 = resize_like(upcnv4, cnv3b)
                i4_in  = tf.concat([upcnv4, cnv3b], axis=3)
                icnv4  = slim.conv2d(i4_in, 128, [3, 3], stride=1, scope='icnv4')

                upcnv3 = slim.conv2d_transpose(icnv4, 64,  [3, 3], stride=2, scope='upcnv3')
                i3_in  = tf.concat([upcnv3, cnv2b], axis=3)
                icnv3  = slim.conv2d(i3_in, 64,  [3, 3], stride=1, scope='icnv3')

                upcnv2 = slim.conv2d_transpose(icnv3, 32,  [3, 3], stride=2, scope='upcnv2')
                i2_in  = tf.concat([upcnv2, cnv1b], axis=3)
                icnv2  = slim.conv2d(i2_in, 32,  [3, 3], stride=1, scope='icnv2')

                upcnv1 = slim.conv2d_transpose(icnv2, 32,  [3, 3], stride=2, scope='upcnv1')
                i1_in  = tf.concat([upcnv1], axis=3)

                icnv1  = slim.conv2d(i1_in, 32,  [3, 3], stride=1, scope='icnv1')

                flow  =  slim.conv2d(icnv1, 2, [3, 3], stride=1,
                    activation_fn=None, normalizer_fn=None, scope='flow')

                w  =  slim.conv2d(icnv1, 2, [3, 3], stride=1,
                    activation_fn=None, normalizer_fn=None, scope='w')

            return flow, w


    def compute_loss(self, dP, depth, intrinsics):

        cfg = self.cfg
        height, width = depth.get_shape().as_list()[1:3]
        batch, frames = dP.get_shape().as_list()[0:2]

        motion_loss = 0.0
        for i, dG in enumerate(self.dGs):
            dG_i = tf.reshape(dG[:, 1:], [batch*(frames-1), 4, 4])
            dP_i = tf.reshape(dP[:, 1:], [batch*(frames-1), 4, 4])

            # replicate the keyframe depth and intrinsics across all pairs
            zdepth = tf.reshape(tf.tile(depth,
                [1, frames-1, 1, 1]), [batch*(frames-1), height, width, 1])

            kvec = tf.reshape(tf.tile(intrinsics,
                [1, frames-1]), [batch*(frames-1), 4])

            geoerr, train_metrics = losses.motion_error(dP_i, dG_i, zdepth, kvec)
            motion_loss += geoerr

        for key in train_metrics:
            tf.summary.scalar("motion_%s"%key, train_metrics[key])


        if cfg.FLOWSE3.ITER_COUNT > 0:
            extra = \
                cfg.TRAIN.WEIGHT_WEIGHT * losses.compute_weights_reg_loss(self.weights) + \
                cfg.TRAIN.RESIDUAL_WEIGHT * tf.reduce_mean(self.resids)
        else:
            extra = 0.0

        motion_loss = motion_loss + extra
        return motion_loss

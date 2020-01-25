import tensorflow as tf
import numpy as np
slim = tf.contrib.slim

from .networks.pose_regressors import pose_regressor_factory
from .networks.layer_ops import *

from geometry.transformation import *
from geometry.intrinsics import *
from utils.bilinear_sampler import bilinear_sampler

EPS = 1e-5
MIN_DEPTH = 0.1
MAX_ERROR = 100.0


def compute_weights_reg_loss(ws, k=2048):
    """ Encourages top k weights to be larger """
    iters, batch, frames, ht, wd, _ = tf.unstack(tf.shape(ws), num=6)
    ws = tf.reshape(ws, [iters*batch*frames, ht, wd, 2])
    ws = tf.transpose(ws, [0, 3, 1, 2])
    ws = tf.reshape(ws, [-1, ht * wd])

    top, _ = tf.nn.top_k(ws, k=k, sorted=False, name='topk')
    ref = tf.ones_like(top)
    l = tf.nn.sigmoid_cross_entropy_with_logits(labels=ref, logits=top)
    return tf.reduce_mean(l)


class MotionNetwork:
    def __init__(self, cfg, 
                 reuse=False,
                 schedule=None, 
                 use_regressor=True,
                 is_calibrated=True,
                 bn_is_training=False,
                 is_training=True,
                 mode='keyframe', 
                 scope='Motion'):
        
        self.cfg = cfg
        self.reuse = reuse
        self.mode = mode
        self.scope = scope

        self.is_calibrated = cfg.IS_CALIBRATED
        if not is_calibrated:
            self.is_calibrated = is_calibrated

        self.is_training = is_training
        self.mode = mode
        self.use_regressor = use_regressor

        self.transform_history = []
        self.coords_history = []
        self.residual_history = []
        self.inds_history = []
        self.weights_history = []
        self.intrinsics_history = []
        self.summaries = []

        self.batch_norm_params = {
            'decay': .995,
            'epsilon': 1e-5,
            'trainable': bn_is_training,
            'is_training': bn_is_training,
        }

    def __len__(self):
        return len(self.transform_history)

    def _all_pairs_indicies(self, num):
        ii, jj = tf.meshgrid(tf.range(num), tf.range(num))
        ii = tf.reshape(ii, [-1])
        jj = tf.reshape(jj, [-1])
        return ii, jj

    def _keyframe_pairs_indicies(self, num):
        ii, jj = tf.meshgrid(tf.range(1), tf.range(1, num))
        ii = tf.reshape(ii, [-1])
        jj = tf.reshape(jj, [-1])
        return ii, jj

    def pose_regressor_init(self, images):
        cfg = self.cfg
        batch, frames = [tf.shape(images)[i] for i in range(2)]

        if not cfg.RESCALE_IMAGES:
            images = images / 255.0

        keyframe, video = images[:, 0], images[:, 1:]
        pose_mat = pose_regressor_factory(keyframe, video, cfg)
        pose_mat = tf.reshape(pose_mat, [batch, frames-1, 4, 4])

        Tij = VideoSE3Transformation(matrix=pose_mat)
        Ts = Tij.append_identity()

        self.transform_history.append(Ts)
        self.residual_history.append(0.0)

        if self.is_training:
            delta_upsilon = cfg.TRAIN.DELTA * tf.random.normal([batch, frames, 6])
            Ts = VideoSE3Transformation(upsilon=delta_upsilon) * Ts

        return Ts

    def extract_features(self, images, reuse=False):
        batch, frames, _, _ = [tf.shape(images)[i] for i in range(4)]
        # images = 2 * (images / 255.0) - 1.0

        height = images.get_shape().as_list()[2]
        width = images.get_shape().as_list()[3]
        inputs = tf.reshape(images, [batch*frames, height, width, 3])

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

        net = tf.reshape(net, [batch, frames, height//4, width//4, 64])
        return net
    
    def flownet(self, fmap1, fmap2, reuse=False):
        inputs = tf.concat([fmap1, fmap2], axis=-1)
        batch, num, height, width = [tf.shape(inputs)[i] for i in range(4)]
        
        height = inputs.get_shape().as_list()[2]
        width = inputs.get_shape().as_list()[3]
        dim = inputs.get_shape().as_list()[-1]

        inputs = tf.reshape(inputs, [batch*num, height, width, dim])

        with tf.variable_scope('flow_net') as sc:
            end_points_collection = sc.original_name_scope + '_end_points'
            with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                                normalizer_fn=None,
                                reuse=reuse,
                                weights_regularizer=slim.l2_regularizer(0.0005),
                                activation_fn=tf.nn.relu,
                                outputs_collections=end_points_collection):

                cnv1  = slim.conv2d(inputs, 32,  [7, 7], stride=2, scope='cnv1')
                tf.add_to_collection("checkpoints", cnv1)

                cnv1b = slim.conv2d(cnv1,  32,  [7, 7], stride=1, scope='cnv1b')
                cnv2  = slim.conv2d(cnv1b, 64,  [5, 5], stride=2, scope='cnv2')
                tf.add_to_collection("checkpoints", cnv1b)

                cnv2b = slim.conv2d(cnv2,  64,  [5, 5], stride=1, scope='cnv2b')
                cnv3  = slim.conv2d(cnv2b, 128, [3, 3], stride=2, scope='cnv3')
                tf.add_to_collection("checkpoints", cnv2b)

                cnv3b = slim.conv2d(cnv3,  128, [3, 3], stride=1, scope='cnv3b')
                cnv4  = slim.conv2d(cnv3b, 256, [3, 3], stride=2, scope='cnv4')
                tf.add_to_collection("checkpoints", cnv3b)

                cnv4b = slim.conv2d(cnv4,  256, [3, 3], stride=1, scope='cnv4b')
                cnv5  = slim.conv2d(cnv4b, 512, [3, 3], stride=2, scope='cnv5')
                tf.add_to_collection("checkpoints", cnv4b)

                cnv5b = slim.conv2d(cnv5,  512, [3, 3], stride=1, scope='cnv5b')

                upcnv5 = slim.conv2d_transpose(cnv5b, 256, [3, 3], stride=2, scope='upcnv5')
                upcnv5 = resize_like(upcnv5, cnv4b)
                i5_in  = tf.concat([upcnv5, cnv4b], axis=3)
                icnv5  = slim.conv2d(i5_in, 256, [3, 3], stride=1, scope='icnv5')
                tf.add_to_collection("checkpoints", icnv5)

                upcnv4 = slim.conv2d_transpose(icnv5, 128, [3, 3], stride=2, scope='upcnv4')
                upcnv4 = resize_like(upcnv4, cnv3b)
                i4_in  = tf.concat([upcnv4, cnv3b], axis=3)
                icnv4  = slim.conv2d(i4_in, 128, [3, 3], stride=1, scope='icnv4')
                tf.add_to_collection("checkpoints", icnv4)

                upcnv3 = slim.conv2d_transpose(icnv4, 64,  [3, 3], stride=2, scope='upcnv3')
                upcnv3 = resize_like(upcnv3, cnv2b)
                i3_in  = tf.concat([upcnv3, cnv2b], axis=3)
                icnv3  = slim.conv2d(i3_in, 64,  [3, 3], stride=1, scope='icnv3')
                tf.add_to_collection("checkpoints", icnv3)

                upcnv2 = slim.conv2d_transpose(icnv3, 32,  [3, 3], stride=2, scope='upcnv2')
                i2_in  = tf.concat([upcnv2, cnv1b], axis=3)
                icnv2  = slim.conv2d(i2_in, 32,  [3, 3], stride=1, scope='icnv2')
                tf.add_to_collection("checkpoints", icnv2)

                upcnv1 = slim.conv2d_transpose(icnv2, 32,  [3, 3], stride=2, scope='upcnv1')
                i1_in  = tf.concat([upcnv1], axis=3)

                icnv1  = slim.conv2d(i1_in, 32,  [3, 3], stride=1, scope='icnv1')

                flow   =  slim.conv2d(icnv1, 2, [3, 3], stride=1,
                    activation_fn=None, normalizer_fn=None, scope='flow')

                weight =  slim.conv2d(icnv1, 2, [3, 3], stride=1,
                    activation_fn=None, normalizer_fn=None, scope='w')

        tf.add_to_collection("checkpoints", flow)
        tf.add_to_collection("checkpoints", weight)

        flow = tf.reshape(flow, [batch, num, height, width, 2])
        weight = tf.reshape(weight, [batch, num, height, width, 2])    
        return flow, weight


    def forward(self, Ts, images, depths, intrinsics, inds=None, num_fixed=0, init=tf.constant(False)):
        # motion network performs projection operations in features space
        cfg = self.cfg
        batch = tf.shape(images)[0]
        num = tf.shape(images)[1]

        if cfg.RESCALE_IMAGES:
            images = 2 * (images / 255.0) - 1.0

        if inds is None:
            if self.mode == 'keyframe':
                self.inds = self._keyframe_pairs_indicies(num)
                num_fixed = 1
            elif self.mode == 'global':
                self.inds = self._all_pairs_indicies(num)
        else:
            self.inds = inds

        (ii, jj) = self.inds
        intrinsics = intrinsics_vec_to_matrix(intrinsics)

        # if self.is_training and (not self.is_calibrated):
        #     perturbation = 0.1 * tf.random.normal([batch, 1])
        #     intrinsics = update_intrinsics(intrinsics, perturbation)

        depths_low, intrinsics = rescale_depths_and_intrinsics(depths, intrinsics, downscale=4)

        with tf.variable_scope("motion", reuse=self.reuse) as sc:
            if Ts is None:
                Ts = self.pose_regressor_init(images)
            else:
                if self.use_regressor:
                    Gs = self.pose_regressor_init(images)
                    Ts = cond_transform(init, Gs, Ts)

            feats = self.extract_features(images)
            depths = tf.gather(depths_low, ii, axis=1) + EPS

            feats1 = tf.gather(feats, ii, axis=1)
            feats2 = tf.gather(feats, jj, axis=1)

            Ti = Ts.gather(ii)
            Tj = Ts.gather(jj)
            Tij = Tj * Ti.inv()

            for i in range(cfg.FLOWSE3.ITER_COUNT):
                Tij = Tij.copy(stop_gradients=True)
                Ts = Ts.copy(stop_gradients=True)
                intrinsics = tf.stop_gradient(intrinsics)

                coords, vmask = Tij.transform(depths, intrinsics, valid_mask=True)
                featsw = vmask * bilinear_sampler(feats2, coords, batch_dims=2)

                with tf.name_scope("residual"):
                    flow, weight = self.flownet(feats1, featsw, reuse=i>0)
                    self.weights_history.append(weight)

                    target = flow + coords
                    weight = vmask * tf.nn.sigmoid(weight)


                with tf.name_scope("PnP"):
                    if (self.mode == 'keyframe') and self.is_calibrated:
                        Tij = Tij.keyframe_optim(target, weight, depths, intrinsics)
                        Ts = Tij.append_identity() # set keyframe pose to identity

                    else:
                        Ts, intrinsics = Ts.global_optim(target, weight, depths, intrinsics, 
                            (jj,ii), num_fixed=num_fixed, include_intrinsics=(not self.is_calibrated))
                        Tij = Ts.gather(jj) * Ts.gather(ii).inv() # relative poses
                    
                    coords, vmask1 = Tij.transform(depths, intrinsics, valid_mask=True)
                    self.transform_history.append(Ts)
                    self.residual_history.append(vmask*vmask1*(coords-target))

                self.intrinsics_history.append(intrinsics_matrix_to_vec(intrinsics))

        intrinsics = 4.0 * intrinsics_matrix_to_vec(intrinsics)
        return Ts, intrinsics


    def compute_loss(self, Gs, depths, intrinsics, loss='l1', log_error=True):
        cfg = self.cfg
        batch, num = Gs.shape()

        ii, jj = self.inds
        intrinsics = intrinsics_vec_to_matrix(intrinsics)

        depths, intrinsics = rescale_depths_and_intrinsics(depths, intrinsics, downscale=4)
        depths = tf.gather(depths, ii, axis=1)

        total_loss = 0.0
        for i in range(len(self.transform_history)):
            Ts = self.transform_history[i]
            Tij = Ts.gather(jj) * Ts.gather(ii).inv()
            Gij = Gs.gather(jj) * Gs.gather(ii).inv()
  
            intrinsics_pred = intrinsics
            if i > 0:
                intrinsics_vec_to_matrix(self.intrinsics_history[i-1])

            zstar = depths + EPS
            flow_pred, valid_mask_pred = Tij.induced_flow(zstar, intrinsics_pred, valid_mask=True) # use predicted intrinsics
            flow_star, valid_mask_star = Gij.induced_flow(zstar, intrinsics, valid_mask=True)
            valid_mask = tf.multiply(valid_mask_pred, valid_mask_star)

            if loss == 'l1':
                reproj_diff = valid_mask * tf.clip_by_value(tf.abs(flow_pred - flow_star), -MAX_ERROR, MAX_ERROR)
                reproj_loss = tf.reduce_mean(reproj_diff)

                w = tf.nn.sigmoid(self.weights_history[i-1])
                resid = tf.clip_by_value(self.residual_history[i], -MAX_ERROR, MAX_ERROR)
                
                resid_loss = tf.reduce_mean(valid_mask * w * resid**2)
                total_loss += reproj_loss + self.cfg.TRAIN.RESIDUAL_WEIGHT * resid_loss

                if log_error:
                    with tf.device('/cpu:0'):
                        self.summaries.append(tf.summary.scalar("reproj_loss", reproj_loss))
                        self.summaries.append(tf.summary.scalar("resid_loss", resid_loss))

            elif loss == 'photometric':
                pass

        # encourage larger weights
        ws = tf.stack(self.weights_history, axis=0)
        weights_loss = compute_weights_reg_loss(ws)
        total_loss += self.cfg.TRAIN.WEIGHT_REG * weights_loss

        return total_loss


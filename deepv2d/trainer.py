import tensorflow as tf
import numpy as np
import os
import cv2

from data_layer import DataLayer, DBDataLayer

from geometry.transformation import *
from utils.memory_saving_gradients import gradients
from utils.average_grads import average_gradients
from utils import mem_util

from modules.depth import DepthNetwork
from modules.motion import MotionNetwork


MOTION_LR_FRACTION = 0.1

class DeepV2DTrainer(object):

    def __init__(self, cfg):
        self.cfg = cfg

    def build_train_graph_stage1(self, cfg, num_gpus=1):

        id_batch, images_batch, poses_batch, gt_batch, filled_batch, pred_batch, intrinsics_batch = self.dl.next()
        images_batch = tf.split(images_batch, num_gpus)
        poses_batch = tf.split(poses_batch, num_gpus)
        gt_batch = tf.split(gt_batch, num_gpus)
        filled_batch = tf.split(filled_batch, num_gpus)
        pred_batch = tf.split(pred_batch, num_gpus)
        intrinsics_batch = tf.split(intrinsics_batch, num_gpus)

        with tf.name_scope("training_schedule"):
            global_step = tf.Variable(0, name='global_step', trainable=False)
            lr = tf.train.exponential_decay(cfg.TRAIN.LR, global_step, 5000, 0.5, staircase=True)
            optim = tf.train.RMSPropOptimizer(MOTION_LR_FRACTION * lr)

        tower_grads = []
        tower_losses = []

        for gpu_id in range(num_gpus):
            images = images_batch[gpu_id]
            poses = poses_batch[gpu_id]
            depth_gt = gt_batch[gpu_id]
            depth_filled = filled_batch[gpu_id]
            depth_pred = pred_batch[gpu_id]
            intrinsics = intrinsics_batch[gpu_id]

            Gs = VideoSE3Transformation(matrix=poses)
            motion_net = MotionNetwork(cfg.MOTION, bn_is_training=True, reuse=gpu_id>0)

            with tf.device('/gpu:%d' % gpu_id):

                depth_input = tf.expand_dims(depth_filled, 1)
                Ts, kvec = motion_net.forward(None, images, depth_input, intrinsics)

                total_loss = motion_net.compute_loss(Gs, depth_input, intrinsics, log_error=(gpu_id==0))
                tower_losses.append(total_loss)

                var_list = tf.trainable_variables()
                grads = gradients(total_loss, var_list)

                gvs = []
                for (g, v) in zip(grads, var_list):
                    if g is not None:
                        if cfg.TRAIN.CLIP_GRADS:
                            g = tf.clip_by_value(g, -1.0, 1.0)
                        gvs.append((g,v))

                gvs = zip(grads, var_list)
                tower_grads.append(gvs)

                # use last gpu to compute batch norm statistics
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)


        with tf.name_scope("train_op"):
            gvs = average_gradients(tower_grads)
            total_loss = tf.reduce_mean(tf.stack(tower_losses, axis=0))

            with tf.control_dependencies(update_ops):
                self.train_op = optim.apply_gradients(gvs, global_step)

            self.write_op = None
            self.total_loss = total_loss
            tf.summary.scalar("learning_rate", lr)
            tf.summary.scalar("total_loss", total_loss)


    def build_train_graph_stage2(self, cfg, num_gpus=1):

        with tf.name_scope("training_schedule"):
            global_step = tf.Variable(0, name='global_step', trainable=False)
            gs = tf.to_float(global_step)
            if cfg.TRAIN.RENORM:
                rmax = tf.clip_by_value(5.0*(gs/2.5e4)+1.0, 1.0, 5.0) # rmax schedule
                dmax = tf.clip_by_value(8.0*(gs/2.5e4), 0.0, 8.0) # dmax schedule
                rmin = 1.0 / rmax
                schedule = {'rmax': rmax, 'rmin': rmin, 'dmax': dmax}
            else:
                schedule = None

            LR_DECAY = int(0.8 * self.training_steps)
            lr = tf.train.exponential_decay(cfg.TRAIN.LR, global_step, LR_DECAY, 0.2, staircase=True)

            stereo_optim = tf.train.RMSPropOptimizer(lr)
            motion_optim = tf.train.RMSPropOptimizer(MOTION_LR_FRACTION*lr)

        id_batch, images_batch, poses_batch, gt_batch, filled_batch, pred_batch, intrinsics_batch = self.dl.next()
        images_batch = tf.split(images_batch, num_gpus)
        poses_batch = tf.split(poses_batch, num_gpus)
        gt_batch = tf.split(gt_batch, num_gpus)
        filled_batch = tf.split(filled_batch, num_gpus)
        pred_batch = tf.split(pred_batch, num_gpus)
        intrinsics_batch = tf.split(intrinsics_batch, num_gpus)

        tower_motion_grads = []
        tower_stereo_grads = []
        tower_predictions = []
        tower_losses = []
        write_ops = []

        for gpu_id in range(num_gpus):
            motion_net = MotionNetwork(cfg.MOTION, reuse=gpu_id>0)
            depth_net = DepthNetwork(cfg.STRUCTURE, schedule=schedule, reuse=gpu_id>0)

            images = images_batch[gpu_id]
            poses = poses_batch[gpu_id]
            depth_gt = gt_batch[gpu_id]
            depth_filled = filled_batch[gpu_id]
            depth_pred = pred_batch[gpu_id]
            intrinsics = intrinsics_batch[gpu_id]

            Gs = VideoSE3Transformation(matrix=poses)
            batch, frames, height, width, _ = images.get_shape().as_list()

            with tf.name_scope("depth_input"):
                input_prob = tf.train.exponential_decay(2.0, global_step, LR_DECAY, 0.02, staircase=False)
                rnd = tf.random_uniform([], 0, 1)
                depth_input = tf.cond(rnd<input_prob, lambda: depth_filled, lambda: depth_pred)

            with tf.device('/gpu:%d' % gpu_id):

                # motion inference
                Ts, kvec = motion_net.forward(None, images, depth_input[:,tf.newaxis], intrinsics)
            
                stop_cond = global_step < cfg.TRAIN.GT_POSE_ITERS
                Ts = cond_transform(stop_cond, Ts.copy(stop_gradients=True), Ts)
                kvec = tf.cond(stop_cond, lambda: tf.stop_gradient(kvec), lambda: kvec)

                # depth inference
                depth_pr = depth_net.forward(Ts, images, kvec)

                depth_loss = depth_net.compute_loss(depth_gt, log_error=(gpu_id==0))
                motion_loss = motion_net.compute_loss(Gs,
                    depth_filled[:,tf.newaxis], intrinsics, log_error=(gpu_id==0))


                # compute all gradients
                if 1:
                    total_loss = cfg.TRAIN.DEPTH_WEIGHT * depth_loss + motion_loss
                    var_list = tf.trainable_variables()
                    grads = gradients(total_loss, var_list)

                # split backward pass
                else:
                    motion_vars = tf.get_collection(tf.GraphKeys.MODEL_VARIABLES, scope="motion")
                    stereo_vars = tf.get_collection(tf.GraphKeys.MODEL_VARIABLES, scope="stereo")

                    so3, translation = Ts.so3, Ts.translation
                    stereo_grads = gradients(depth_loss, [so3, translation] + stereo_vars)
                    diff_so3, diff_translation, stereo_grads = \
                        stereo_grads[0], stereo_grads[1], stereo_grads[2:]

                    motion_grads = tf.gradients([motion_loss, so3, translation], motion_vars, 
                        grad_ys=[tf.ones_like(motion_loss), diff_so3, diff_translation])

                    grads = stereo_grads + motion_grads
                    var_list = stereo_vars + motion_vars

                motion_gvs = []
                stereo_gvs = []

                for (g, v) in zip(grads, var_list):
                    if 'stereo' in v.name and (g is not None):
                        if cfg.TRAIN.CLIP_GRADS:
                            g = tf.clip_by_value(g, -1.0, 1.0)
                        stereo_gvs.append((g,v))

                    if 'motion' in v.name and (g is not None):
                        if cfg.TRAIN.CLIP_GRADS and (g is not None):
                            g = tf.clip_by_value(g, -1.0, 1.0)
                        motion_gvs.append((g,v))

                tower_motion_grads.append(motion_gvs)
                tower_stereo_grads.append(stereo_gvs)

                tower_predictions.append(depth_pr)
                tower_losses.append(depth_loss)

                if gpu_id == 0:
                    self.total_loss = depth_loss

                # use last gpu to compute batch norm statistics
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)


        tower_motion_gvs = average_gradients(tower_motion_grads)
        tower_stereo_gvs = average_gradients(tower_stereo_grads)

        with tf.name_scope("train_op"):
            with tf.control_dependencies(update_ops):
                self.train_op = tf.group(
                    stereo_optim.apply_gradients(tower_stereo_gvs),
                    motion_optim.apply_gradients(tower_motion_gvs),
                    tf.assign(global_step, global_step+1)
                )

        self.write_op = self.dl.write(id_batch, tf.concat(tower_predictions, axis=0))
        self.total_loss = tf.reduce_mean(tf.stack(tower_losses, axis=0))

        tf.summary.scalar("total_loss", self.total_loss)
        tf.summary.scalar("learning_rate", lr)
        tf.summary.scalar("input_prob", input_prob)


    def train(self, data_source, cfg, stage=1, ckpt=None, restore_ckpt=None, num_gpus=1):

        batch_size = num_gpus * cfg.TRAIN.BATCH[stage-1]
        max_steps = cfg.TRAIN.ITERS[stage-1]
        self.training_steps = max_steps

        print ("batch size: %d \t max steps: %d"%(batch_size, max_steps))
        if isinstance(data_source, str):
            self.dl = DataLayer(data_source, batch_size=batch_size)
        else:
            self.dl = DBDataLayer(data_source, batch_size=batch_size)

        if stage == 1:
            self.build_train_graph_stage1(cfg, num_gpus=num_gpus)

        elif stage == 2:
            self.build_train_graph_stage2(cfg, num_gpus=num_gpus)


        self.summary_op = tf.summary.merge_all()

        saver = tf.train.Saver([var for var in tf.model_variables()], max_to_keep=10)
        train_writer = tf.summary.FileWriter(cfg.LOG_DIR+'_stage_%s'%str(stage))

        init_op = tf.group(tf.global_variables_initializer(),
                   tf.local_variables_initializer())

        SUMMARY_FREQ = 10
        LOG_FREQ = 100
        CHECKPOINT_FREQ = 5000

        with tf.Session() as sess:
            sess.run(init_op)

            # train with tfrecords
            if isinstance(self.dl, DataLayer):
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(coord=coord)

            # train with python data loader
            elif isinstance(self.dl, DBDataLayer):
                self.dl.init(sess)

            kwargs = {}

            if stage >= 2:
                motion_vars = tf.get_collection(tf.GraphKeys.MODEL_VARIABLES, scope="motion")
                motion_saver = tf.train.Saver(motion_vars)

                if ckpt is not None:
                    motion_saver.restore(sess, ckpt)

                if restore_ckpt is not None:
                    saver.restore(sess, restore_ckpt)

            running_loss = 0.0
            for step in range(1, max_steps):

                kwargs = {}
                fetches = {}
                fetches['train_op'] = self.train_op
                fetches['loss'] = self.total_loss
                if self.write_op is not None:
                    fetches['write_op'] = self.write_op

                if step % SUMMARY_FREQ == 0:
                    fetches['summary'] = self.summary_op

                result = sess.run(fetches, **kwargs)

                if step % SUMMARY_FREQ == 0:
                    train_writer.add_summary(result['summary'], step)

                if step % LOG_FREQ == 0:
                    print('[stage=%d, %5d] loss: %.3f'%(stage, step, running_loss / LOG_FREQ))
                    running_loss = 0.0

                if step % CHECKPOINT_FREQ == 0:
                    checkpoint_file = os.path.join(cfg.CHECKPOINT_DIR, '_stage_%s.ckpt'%str(stage))
                    saver.save(sess, checkpoint_file, step)

                running_loss += result['loss']

            checkpoint_file = os.path.join(cfg.CHECKPOINT_DIR, '_stage_%s.ckpt'%str(stage))
            saver.save(sess, checkpoint_file)

        return checkpoint_file

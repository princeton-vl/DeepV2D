import tensorflow as tf
import numpy as np
import os
import cv2

from utils import metrics
import se3
from data_layer import DataLayer

from utils.memory_saving_gradients import gradients
from networks.structure import StructureNetwork
from networks.motion import MotionNetwork


class DeepV2DTrainer(object):

    def __init__(self, cfg):
        self.cfg = cfg

    def build_train_graph_stage1(self, cfg):

        id, images, poses, depth_gt, filled, depth_pred, intrinsics = self.dl.next()
        images = tf.to_float(images)
        batch, frames, height, width, _ = images.get_shape().as_list()

        with tf.name_scope("training_schedule"):
            global_step = tf.Variable(0, name='global_step', trainable=False)
            gs = tf.to_float(global_step)
            schedule = {'rmax': 1.0, 'rmin': 1.0, 'dmax': 0.0}

            lr = tf.train.exponential_decay(cfg.TRAIN.LR, global_step, 5000, 0.5, staircase=True)
            optim = tf.train.RMSPropOptimizer(0.2 * lr)

        dPs = []
        for i in range(frames):
            dPs.append(se3.solve_SE3(poses[:, 0], poses[:, i]))
        dP = tf.stack(dPs, axis=1)

        net = MotionNetwork(cfg.MOTION, schedule=schedule)
        dG = net.forward(images[:, 1:], images[:, 0], filled, intrinsics)

        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.name_scope("train_op"):
            total_loss = net.compute_loss(dP, depth_gt, intrinsics)
            var_list = tf.trainable_variables()
            grads = gradients(total_loss, var_list)

            gvs = []
            for (g, v) in zip(grads, var_list):
                if cfg.TRAIN.CLIP_GRADS and (g is not None):
                    g = tf.clip_by_value(g, -1.0, 1.0)
                gvs.append((g,v))

            gvs = zip(grads, var_list)

            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                self.train_op = optim.apply_gradients(gvs, global_step)

        self.write_op = None
        self.total_loss = total_loss
        tf.summary.scalar("learning_rate", lr)
        tf.summary.scalar("total_loss", total_loss)



    def build_train_graph_stage2(self, cfg):

        id, images, poses, depth_gt, filled, depth_pred, intrinsics = self.dl.next()
        images = tf.to_float(images)
        batch, frames, height, width, _ = images.get_shape().as_list()

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

            LR_DECAY = 100000
            lr = tf.train.exponential_decay(cfg.TRAIN.LR, global_step, LR_DECAY, 0.2, staircase=True)

            stereo_optim = tf.train.RMSPropOptimizer(lr)
            motion_optim = tf.train.RMSPropOptimizer(0.1*lr) # use lower learning rate for motion network

        dPs = []
        for i in range(frames):
            dPs.append(se3.solve_SE3(poses[:, 0], poses[:, i]))
        dP = tf.stack(dPs, axis=1)

        motion_net = MotionNetwork(cfg.MOTION, schedule=schedule)
        stereo_net = StructureNetwork(cfg.STRUCTURE, schedule=schedule)

        with tf.name_scope("depth_input"):
            input_prob = tf.train.exponential_decay(2.0, global_step, LR_DECAY, 0.02, staircase=False)
            rnd = tf.random_uniform([], 0, 1)
            depth_input = tf.cond(rnd<input_prob, lambda: filled, lambda: depth_pred)


        dG = motion_net.forward(images[:, 1:], images[:, 0], depth_input, intrinsics)
        # use gt pose for first iterations
        dG = tf.cond(global_step<cfg.TRAIN.GT_POSE_ITERS, lambda: dP, lambda: dG)
        stereo_depth = stereo_net.forward(images, dG, intrinsics)

        train_metrics = metrics.eval(depth_gt, tf.expand_dims(stereo_depth, -1))
        for key in train_metrics:
            tf.summary.scalar("depth_%s"%key, train_metrics[key])

        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.name_scope("train_op"):
            stereo_loss = stereo_net.compute_loss(depth_gt)
            motion_loss = motion_net.compute_loss(dP, depth_gt, intrinsics)
            total_loss = stereo_loss + motion_loss

            var_list = tf.trainable_variables()
            grads = gradients(total_loss, var_list)

            motion_gvs = []
            stereo_gvs = []
            for (g, v) in zip(grads, var_list):
                if 'stereo' in v.name:
                    if cfg.TRAIN.CLIP_GRADS and (g is not None):
                        g = tf.clip_by_value(g, -1.0, 1.0)
                    stereo_gvs.append((g,v))
                if 'motion' in v.name:
                    if cfg.TRAIN.CLIP_GRADS and (g is not None):
                        g = tf.clip_by_value(g, -1.0, 1.0)
                    motion_gvs.append((g,v))

            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                self.train_op = tf.group(
                    stereo_optim.apply_gradients(stereo_gvs),
                    motion_optim.apply_gradients(motion_gvs),
                    tf.assign(global_step, global_step+1)
                )

        self.write_op = self.dl.write(id, stereo_depth)
        self.total_loss = stereo_loss
        tf.summary.scalar("learning_rate", lr)
        tf.summary.scalar("total_loss", total_loss)
        tf.summary.scalar("input_prob", input_prob)


    def train(self, tfrecords, cfg, stage=1, ckpt=None):

        batch_size = cfg.TRAIN.BATCH[stage-1]
        max_steps = cfg.TRAIN.ITERS[stage-1]

        print ("batch size: %d \t max steps: %d"%(batch_size, max_steps))
        self.dl = DataLayer(tfrecords, batch_size=batch_size)

        if stage == 1:
            self.build_train_graph_stage1(cfg)

        elif stage == 2:
            self.build_train_graph_stage2(cfg)

        self.summary_op = tf.summary.merge_all()

        saver = tf.train.Saver([var for var in tf.model_variables()], max_to_keep=10)
        train_writer = tf.summary.FileWriter(cfg.LOG_DIR+'_stage_%s'%str(stage))

        init_op = tf.group(tf.global_variables_initializer(),
                   tf.local_variables_initializer())

        SUMMARY_FREQ = 10
        LOG_FREQ = 100
        CHECKPOINT_FREQ = 10000

        with tf.Session() as sess:
            sess.run(init_op)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            kwargs = {}

            if stage == 2:
                motion_saver = tf.train.Saver(tf.get_collection(
                    tf.GraphKeys.MODEL_VARIABLES, scope="motion"))
                motion_saver.restore(sess, ckpt)

            running_loss = 0.0
            for step in range(1, max_steps):

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


                running_loss += result['loss']

            checkpoint_file = os.path.join(cfg.CHECKPOINT_DIR, '_stage_%s.ckpt'%str(stage))
            saver.save(sess, checkpoint_file)

        return checkpoint_file

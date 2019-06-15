import tensorflow as tf
import numpy as np

from networks.structure import StructureNetwork
from networks.motion import MotionNetwork
from fcrn import fcrn
import se3


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


class DeepV2DSLAM(object):

    def __init__(self, dims, cfg, use_fcrn=False):
        self.dims = dims
        self.cfg = cfg
        frames, height, width = self.dims

        # video clip for depth estimation
        keyframe = tf.placeholder(tf.float32, [1, height, width, 3])
        images = tf.placeholder(tf.float32, [1, frames+1, height, width, 3])
        poses = tf.placeholder(tf.float32, [1, frames+1, 4, 4])

        # image pair for tracking
        image_pair = tf.placeholder(tf.float32, [1, 2, height, width, 3])
        pose_pair = tf.placeholder(tf.float32, [1, 2, 4, 4])

        depth = tf.placeholder(tf.float32, [1, height, width, 1])
        poses = tf.placeholder(tf.float32, [1, frames+1, 4, 4])
        intrinsics = tf.placeholder(tf.float32, [1, 4])

        self.placeholders = {
            'keyframe': keyframe,
            'images': images,
            'image_pair': image_pair,
            'depth': depth,
            'poses': poses,
            'pose_pair': pose_pair,
            'intrinsics': intrinsics,
        }

        self.outputs = {}
        self._build_depth_graph()
        self._build_motion_graph()
        self._build_fcrn_graph()


        ###### SLAM variables #####
        self.images = []
        self.poses = []

        self.keyframe_index = 0
        self.keyframe_image = None
        self.keyframe_depth = None


        ###### SLAM parameters #####
        self.n_frames = self.dims[0]
        self.KEYFRAME_STEP = 3
        self.FORWARD_MARGIN = 6
        self.BACWARD_MARGIN = 8

        self.DO_POSE_UPDATE = True # update the pose estimates after tracking


    def _build_depth_graph(self):
        frames, height, width = self.dims
        images = self.placeholders['images']
        poses = self.placeholders['poses']
        intrinsics = self.placeholders['intrinsics']

        delta_poses = []
        for i in range(frames):
            delta_poses.append(se3.solve_SE3(poses[:,0], poses[:,i]))
        delta_poses = tf.stack(delta_poses, axis=1)

        stereo = StructureNetwork(self.cfg.STRUCTURE, is_training=False)
        depth = stereo.forward(images, delta_poses, intrinsics)
        self.outputs['depth'] = tf.expand_dims(depth[0], axis=-1)


    def _build_motion_graph(self):
        frames, height, width = self.dims
        images = self.placeholders['image_pair']
        poses = self.placeholders['pose_pair']
        depth = self.placeholders['depth']
        intrinsics = self.placeholders['intrinsics']

        motion = MotionNetwork(self.cfg.MOTION, is_training=False)
        delta_pose = tf.matmul(poses[:,1], tf.linalg.inv(poses[:,0]))
        
        # se3.solve_SE3(poses[:,0], poses[:,1])[:, tf.newaxis]
        dP = motion.forward(images[:, 1:], images[:, 0], depth, intrinsics, G0=delta_pose)

        # update pose
        pose = tf.matmul(dP[:,1], poses[:,0])
        self.outputs['pose'] = pose[0]


    def _build_fcrn_graph(self):
        image = self.placeholders['keyframe']
        batch, height, width, _ = image.get_shape().as_list()

        with tf.variable_scope("FCRN") as scope:
            # crop out boarder and flip color channels
            fcrn_input = tf.image.resize_area(image[:, 4:-4, 6:-6, ::-1], [228, 304])
            net = fcrn.ResNet50UpProj({'data': fcrn_input}, batch, 1, False)
            dpred = tf.stop_gradient(net.get_output())
            dpred = tf.image.resize_bilinear(dpred, [height, width])

        self.outputs['fcrn'] = dpred[0]


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


    def set_intrinsics(self, intrinsics):
        self.intrinsics = intrinsics


    def update(self, image):

        ########## initialize the tracker ##########
        if len(self.images) == 0:
            self.keyframe_image = image
            self.keyframe_pose = np.eye(4)
            self.images.append(image)
            self.poses.append(self.keyframe_pose.copy())

            feed_dict = {
                self.placeholders['keyframe']: image[np.newaxis]
            }

            # initialize depth map using fcrn
            depth = self.sess.run(self.outputs['fcrn'], feed_dict=feed_dict)
            self.keyframe_depth = depth

        else:

            ########## track new image ##########
            image_pair = np.stack([self.keyframe_image, image], axis=0)
            poses_pair = np.stack([self.keyframe_pose, self.poses[-1]], axis=0)

            feed_dict = {
                self.placeholders['image_pair']: image_pair[np.newaxis],
                self.placeholders['pose_pair']: poses_pair[np.newaxis],
                self.placeholders['depth']: self.keyframe_depth[np.newaxis],
                self.placeholders['intrinsics']: self.intrinsics[np.newaxis],
            }

            pose_new = self.sess.run(self.outputs['pose'], feed_dict=feed_dict)

            self.images.append(image)
            self.poses.append(pose_new)

             ########## keyframe update ##########
            if len(self.poses) - self.keyframe_index > self.FORWARD_MARGIN:
                self.keyframe_index += self.KEYFRAME_STEP
                self.keyframe_image = self.images[self.keyframe_index].copy()
                self.keyframe_pose = self.poses[self.keyframe_index].copy()


            ########## depth update ##########
            images = [self.keyframe_image]
            poses = [self.keyframe_pose]

            ixs = range(max(0,self.keyframe_index-self.BACWARD_MARGIN), self.keyframe_index) + \
                  range(self.keyframe_index+1, len(self.images))

            if len(ixs) > 2:
                if len(ixs) < self.n_frames:
                    ixs = np.random.choice(ixs, size=self.n_frames, replace=True)
                else:
                    ixs = np.random.choice(ixs, size=self.n_frames, replace=False)
           
                for i in ixs:
                    images.append(self.images[i])
                    poses.append(self.poses[i])

                images = np.stack(images, axis=0)
                poses = np.stack(poses, axis=0)

                feed_dict = {
                    self.placeholders['images']: images[np.newaxis],
                    self.placeholders['poses']: poses[np.newaxis],
                    self.placeholders['intrinsics']: self.intrinsics[np.newaxis],
                }

                depth_new = self.sess.run(self.outputs['depth'], feed_dict=feed_dict)
                self.keyframe_depth = depth_new

            # ########## pose update ##########
            if self.DO_POSE_UPDATE:
                for i in range(self.keyframe_index+1, len(self.poses)): # only update poses after keyframe
                    image_pair = np.stack([self.keyframe_image, self.images[i]], axis=0)
                    poses_pair = np.stack([self.keyframe_pose, self.poses[i]], axis=0)

                    feed_dict = {
                        self.placeholders['image_pair']: image_pair[np.newaxis],
                        self.placeholders['pose_pair']: poses_pair[np.newaxis],
                        self.placeholders['depth']: self.keyframe_depth[np.newaxis],
                        self.placeholders['intrinsics']: self.intrinsics[np.newaxis],
                    }

                    pose_new = self.sess.run(self.outputs['pose'], feed_dict=feed_dict)
                    self.poses[i] = pose_new
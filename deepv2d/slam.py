import tensorflow as tf
import numpy as np
import time
from multiprocessing import Process, Queue

from modules.depth import DepthNetwork
from modules.motion import MotionNetwork

from fcrn import fcrn
import vis
import cv2

from geometry.transformation import *
from geometry.intrinsics import *
from geometry import projective_ops
import matplotlib.pyplot as plt
from scipy import interpolate


def fill_depth(depth):
    """ Fill in the holes in the depth map """
    ht, wd = depth.shape
    x, y = np.meshgrid(np.arange(wd), np.arange(ht))
    xx = x[depth > 0].astype(np.float32)
    yy = y[depth > 0].astype(np.float32)
    zz = depth[depth > 0].ravel()
    return interpolate.griddata((xx, yy), zz, (x, y), method='nearest')

def vee(R):
    x1 = R[2,1] - R[1,2]
    x2 = R[0,2] - R[2,0]
    x3 = R[1,0] - R[0,1]
    return np.array([x1, x2, x3])

def pose_distance(G):
    R, t = G[:3,:3], G[:3,3]
    r = vee(R)
    dR = np.sqrt(np.sum(r**2))
    dt = np.sqrt(np.sum(t**2))
    return dR + dt


class DeepV2DSLAM:
    def __init__(self, cfg, ckpt, n_keyframes=1, rate=2, use_fcrn=True, 
            viz=True, mode='global', image_dims=[None, 480, 640]):
        
        self.cfg = cfg
        self.ckpt = ckpt

        self.viz = viz
        self.mode = mode
        self.use_fcrn = use_fcrn
        self.image_dims = image_dims

        self.index = 0
        self.keyframe_inds = []

        self.images = []
        self.depths = []
        self.poses = []

        # tracking config parameters
        self.n_keyframes = n_keyframes # number of keyframes to use
        self.rate = rate # how often to sample new frames
        self.window = 3  # add edges if frames are within distance

        # build tensorflow graphs
        self.outputs = {}
        self._create_placeholders()
        self._build_motion_graph()
        self._build_depth_graph()
        self._build_reprojection_graph()
        self._build_visibility_graph()
        self._build_point_cloud_graph()

        if self.use_fcrn:
            self._build_fcrn_graph()

        self.saver = tf.train.Saver(tf.model_variables())

    def set_session(self, sess):
        self.sess = sess
        self.saver.restore(self.sess, self.ckpt)

        if self.use_fcrn:
            fcrn_vars = {}
            for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="FCRN"):
                fcrn_vars[var.name.replace('FCRN/', '').replace(':0', '')] = var

            fcrn_saver = tf.train.Saver(fcrn_vars)
            fcrn_saver.restore(sess, 'models/NYU_FCRN.ckpt')

    def start_visualization(self, cinematic=False, render_path=None, clear_points=False):
        """ Start interactive slam visualization in seperate process """

        # new points and poses get added to the queue
        self.queue = Queue()
        self.vis_counter = 0
        
        self.viz = vis.InteractiveViz(self.queue, cinematic, render_path, clear_points)
        self.viz.start()


    def _create_placeholders(self):
        frames, ht, wd = self.image_dims
        self.images_placeholder = tf.placeholder(tf.float32, [frames, ht, wd, 3])
        if self.mode == 'keyframe':
            self.depths_placeholder = tf.placeholder(tf.float32, [1, ht, wd])
        else:
            self.depths_placeholder = tf.placeholder(tf.float32, [frames, ht, wd])

        self.poses_placeholder = tf.placeholder(tf.float32, [frames, 4, 4])
        self.intrinsics_placeholder = tf.placeholder(tf.float32, [4])

        # placeholders for storing graph adj_list and edges
        self.edges_placeholder = tf.placeholder(tf.int32, [None, 2])
        self.adj_placeholder = tf.placeholder(tf.int32, [None, None])
        self.fixed_placeholder = tf.placeholder(tf.int32, [])
        self.init_placeholder = tf.placeholder(tf.bool, [])

    def _build_motion_graph(self):
        """ Motion graph updates poses using depth as input """

        self.motion_net = MotionNetwork(self.cfg.MOTION, 
                                        mode='global',          # use global optimization mode
                                        is_training=False)
        
        images = self.images_placeholder[tf.newaxis]
        depths = self.depths_placeholder[tf.newaxis]
        poses = self.poses_placeholder[tf.newaxis]
        intrinsics = self.intrinsics_placeholder[tf.newaxis]
        edge_inds = tf.unstack(self.edges_placeholder, num=2, axis=-1)
            
        # convert pose matricies into SE3 object
        Ts = VideoSE3Transformation(matrix=poses)
        batch, num = Ts.shape()
        
        Ts, intrinsics = self.motion_net.forward(Ts, images, depths, intrinsics, 
                                                 inds=edge_inds, 
                                                 num_fixed=self.fixed_placeholder)

        # convert SE3 object back to matrix representation
        self.outputs['poses'] = tf.squeeze(Ts.matrix(), 0)
        self.outputs['intrinsics'] = intrinsics


    def _build_depth_graph(self):
        """ Depth graph updates depth using poses as input """
        self.depth_net = DepthNetwork(self.cfg.STRUCTURE, is_training=False)
        images = self.images_placeholder[tf.newaxis]
        poses = self.poses_placeholder[tf.newaxis]
        intrinsics = self.intrinsics_placeholder[tf.newaxis]

        Ts = VideoSE3Transformation(matrix=poses)

        adj_list = None
        if self.mode == 'global':
            adj_list = self.adj_placeholder
        
        depths = self.depth_net.forward(Ts, images, intrinsics, adj_list)
        self.outputs['depths'] = depths

    def _build_visibility_graph(self):
        depths = self.depths_placeholder[tf.newaxis]
        poses = self.poses_placeholder[tf.newaxis]
        intrinsics = self.intrinsics_placeholder[tf.newaxis]

        Ts = VideoSE3Transformation(matrix=poses)
        ii, jj = tf.unstack(self.edges_placeholder, num=2, axis=-1)
        intrinsics = intrinsics_vec_to_matrix(intrinsics)

        depths, intrinsics = rescale_depths_and_intrinsics(depths, intrinsics, downscale=4)
        ht = tf.cast(tf.shape(depths)[2], tf.float32)
        wd = tf.cast(tf.shape(depths)[3], tf.float32)

        depths = tf.gather(depths, ii, axis=1)
        Tij = Ts.gather(jj) * Ts.gather(ii).inv()

        flow = Tij.induced_flow(depths, intrinsics)
        coords = Tij.transform(depths, intrinsics)

        # translation only 
        rotation_mask = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
        flow_translation = Tij.induced_flow(depths, intrinsics)

        flo_graph = tf.sqrt(tf.reduce_sum(flow**2, axis=-1))
        flo_graph = tf.reduce_mean(flo_graph, [-1, -2])

        pos_graph = tf.sqrt(tf.reduce_sum(flow_translation**2, axis=-1))
        pos_graph = tf.reduce_mean(pos_graph, [-1, -2])

        contained = tf.to_float(
            (coords[...,0] > 0.0) & (coords[...,0] < wd) & 
            (coords[...,1] > 0.0) & (coords[...,1] < ht))
        
        vis_graph = tf.reduce_mean(contained, [-1, -2])
        self.outputs['visibility'] = (flo_graph[0], vis_graph[0])

    def _build_fcrn_graph(self):
        """ Build single image initializion graph"""
        images = self.images_placeholder
        batch, ht, wd, _ = tf.unstack(tf.shape(images), num=4)

        with tf.variable_scope("FCRN") as scope:
            # crop out boarder and flip color channels
            fcrn_input = tf.image.resize_area(images[:, 4:-4, 6:-6, ::-1], [228, 304])
            net = fcrn.ResNet50UpProj({'data': fcrn_input}, batch, 1, False)
            fcrn_output = tf.stop_gradient(net.get_output())
            fcrn_output = tf.image.resize_bilinear(fcrn_output, [ht, wd])

        self.outputs['fcrn'] = tf.squeeze(fcrn_output, -1)

    def compute_visibility_graph(self, edges=None):
        """ Computes a matrix of optical flow and visibility between all pairs of frames 
        Ex. flo_matrix[i,j] is the mean optical flow between camera i and camera j
        Ex. vis_matrix[i,j] is the portion of points in camera i visibile in camera j """
        
        vis_matrix = False
        if edges is None:
            num = len(self.keyframe_images)
            vis_matrix = True
            ii, jj = np.meshgrid(np.arange(num), np.arange(num))
            
            ii = np.reshape(ii, [-1])
            jj = np.reshape(jj, [-1])
            edges = np.stack([jj, ii], axis=-1)

        feed_dict = {
            self.depths_placeholder: np.stack(self.keyframe_depths, axis=0),
            self.poses_placeholder: np.stack(self.keyframe_poses, axis=0),
            self.edges_placeholder: edges,
            self.intrinsics_placeholder: self.intrinsics}

        flo_graph, pos_graph = self.sess.run(self.outputs['visibility'], feed_dict=feed_dict)
        if vis_matrix:
            flo_matrix = flo_graph.reshape(num, num)
            pos_matrix = pos_graph.reshape(num, num)
            return flo_matrix, pos_matrix

        return flo_graph, pos_matrix

    def _build_point_cloud_graph(self):
        """Use poses and depth maps to create point cloud"""
        depths = self.depths_placeholder[tf.newaxis]
        images = self.images_placeholder[tf.newaxis]
        poses = self.poses_placeholder[tf.newaxis]
        intrinsics = self.intrinsics_placeholder[tf.newaxis]
        intrinsics = intrinsics_vec_to_matrix(intrinsics)

        depths_pad = tf.pad(depths, [[0,0],[0,0],[0,1],[0,1]], "CONSTANT")

        depths_grad = \
            (depths_pad[:, :, 1:, :-1] - depths_pad[:, :, :-1, :-1])**2 + \
            (depths_pad[:, :, :-1, 1:] - depths_pad[:, :, :-1, :-1])**2

        # don't use large depths for point cloud and ignore boundary regions
        valid = (depths < 6.0) & (depths_grad < 0.05)

        batch, num, ht, wd = tf.unstack(tf.shape(depths), num=4)
        Ts = VideoSE3Transformation(matrix=poses)
        X0 = projective_ops.backproject(depths, intrinsics)
        
        # transform point cloud into world coordinaes
        X1 = Ts.inv()(X0)

        crop_h0 = 20
        crop_h1 = 12

        crop_w = 32

        X1 = X1[:, :, crop_h0:-crop_h1, crop_w:-crop_w]
        valid = valid[:, :, crop_h0:-crop_h1, crop_w:-crop_w]
        images = images[:, :, crop_h0:-crop_h1, crop_w:-crop_w, ::-1]
        
        X1 = tf.reshape(X1, [-1, 3])
        colors = tf.reshape(images, [-1, 3])

        valid_inds = tf.where(tf.reshape(valid, [-1]))
        valid_inds = tf.reshape(valid_inds, [-1])

        X1 = tf.gather(X1, valid_inds, axis=0)
        colors = tf.gather(colors, valid_inds, axis=0)

        self.outputs['point_cloud'] = (X1, colors)


    def _build_reprojection_graph(self):
        """ Used to project depth from keyframes onto new frame """
        EPS = 1e-8
        depths = self.depths_placeholder[tf.newaxis]
        poses = self.poses_placeholder[tf.newaxis]
        intrinsics = self.intrinsics_placeholder[tf.newaxis]

        batch, num, ht, wd = tf.unstack(tf.shape(depths), num=4)
        Ts = VideoSE3Transformation(matrix=poses)
        intrinsics = intrinsics_vec_to_matrix(intrinsics)

        ii, jj = tf.meshgrid(tf.range(0, num), tf.range(num, num+1))
        ii = tf.reshape(ii, [-1])
        jj = tf.reshape(jj, [-1])

        Tij = Ts.gather(jj) * Ts.gather(ii).inv()
        X0 = projective_ops.backproject(depths, intrinsics)
        X1 = Tij(X0)

        coords = projective_ops.project(X1, intrinsics)
        depths = X1[..., 2]

        indicies = tf.cast(coords[..., ::-1] + .5, tf.int32)
        indicies = tf.reshape(indicies, [-1, 2])
        depths = tf.reshape(depths, [-1])

        depth = tf.scatter_nd(indicies, depths, [ht, wd])
        count = tf.scatter_nd(indicies, tf.ones_like(depths), [ht, wd])

        depth = depth / (count + EPS)
        self.outputs['depth_reprojection'] = depth


    def reproject_depth(self, query_pose, margin=2):
        """ Use depth estimates and poses to estimate depth map at a new camera location """

        keyframe_pose = self.poses[self.keyframe_inds[-1]]
        poses = np.stack([keyframe_pose, query_pose], axis=0)

        keyframe_depth = self.depths[self.keyframe_inds[-1]]
        depths = keyframe_depth[np.newaxis]

        feed_dict = {
            self.depths_placeholder: depths,
            self.poses_placeholder: poses,
            self.intrinsics_placeholder: self.intrinsics}

        depth = self.sess.run(self.outputs['depth_reprojection'], feed_dict=feed_dict)
        return fill_depth(depth)

    def deepv2d_init(self):
        if self.use_fcrn:
            feed_dict = {self.images_placeholder: np.stack(self.images, axis=0)}
            depths_init = self.sess.run(self.outputs['fcrn'], feed_dict=feed_dict)

        else:
            ii = np.arange(len(self.images))
            adj = np.stack([ii, ii], axis=-1)

            feed_dict = {
                self.images_placeholder: np.stack(self.images, axis=0),
                self.poses_placeholder: np.stack(self.poses, axis=0),
                self.adj_placeholder: adj,
                self.intrinsics_placeholder: self.intrinsics}

            depths_init = self.sess.run(self.outputs['depths'], feed_dict=feed_dict)

        self.depths = [depth for depth in depths_init]

    
    def update_poses(self, fixed=1, margin=3):
        """ Update the poses by executing the motion graph, fix first keyframe """

        n_images = len(self.images)
        start_idx = max(self.keyframe_inds[0] - margin, 0)

        edges = []
        for i in self.keyframe_inds:
            for j in range(start_idx, n_images):
                if (i != j) and (abs(i - j) <= self.window):
                    edges.append((i, j))

        edges = np.stack(edges, axis=0) - start_idx
        images = np.stack(self.images[start_idx:], axis=0)
        depths = np.stack(self.depths[start_idx:], axis=0)
        poses = np.stack(self.poses[start_idx:], axis=0)

        if not fixed:
            fixed = 0

        feed_dict = {
            self.images_placeholder: images,
            self.depths_placeholder: depths,
            self.poses_placeholder: poses,
            self.edges_placeholder: edges,
            self.fixed_placeholder: np.int32(fixed),
            self.init_placeholder: False,
            self.intrinsics_placeholder: self.intrinsics}
            
        # execute pose subgraph
        poses = self.sess.run(self.outputs['poses'], feed_dict=feed_dict)

        # update the poses 
        for j in range(poses.shape[0]):
            self.poses[start_idx + j] = poses[j]

        self.pose_cur = self.poses[-1]


    def update_depths(self, fixed=1, margin=3):
        """ Update the depths by executing the depth graph """

        n_images = len(self.images)
        start_idx = max(self.keyframe_inds[0] - margin, 0)

        # faster if we batch multiple depth updates together
        inds = self.keyframe_inds
        if fixed and len(self.keyframe_inds) > 1:
            inds = inds[fixed:] # fix depth for first keyframe

        adj_list = []
        for i in inds:
            adj_inds = []
            for j in range(start_idx, n_images):
                if (i != j) and (abs(i - j) <= self.window):
                    adj_inds.append(j)
            
            # make sure all adj lists are the same size
            if len(adj_inds) < 2*self.window:
                adj_inds = np.random.choice(adj_inds, 2*self.window, replace=True).tolist()
                
            adj_inds = [i] + adj_inds
            adj_list.append(np.array(adj_inds, dtype=np.int32))

        adj_list = np.stack(adj_list, axis=0) - start_idx
        images = np.stack(self.images[start_idx:], axis=0)
        poses = np.stack(self.poses[start_idx:], axis=0)

        feed_dict = {
            self.images_placeholder: images,
            self.poses_placeholder: poses,
            self.adj_placeholder: adj_list,
            self.intrinsics_placeholder: self.intrinsics,
        }

        depths = self.sess.run(self.outputs['depths'], feed_dict=feed_dict)
        
        # update the keyframe depths
        for i, keyframe_index in enumerate(inds):
            self.depths[keyframe_index] = depths[i]


    def visualize_output(self, keyframe_index):
        """ Backproject a point cloud then add point cloud to visualization """

        self.vis_counter += 1
        keyframe_image = self.images[keyframe_index]
        keyframe_depth = self.depths[keyframe_index]
        keyframe_pose = self.poses[keyframe_index]

        feed_dict = {
            self.images_placeholder: keyframe_image[np.newaxis],
            self.depths_placeholder: keyframe_depth[np.newaxis],
            self.poses_placeholder: keyframe_pose[np.newaxis],
            self.intrinsics_placeholder: self.intrinsics}

        keyframe_point_cloud, keyframe_point_colors = \
            self.sess.run(self.outputs['point_cloud'], feed_dict=feed_dict)

        pointcloud = (keyframe_point_cloud, keyframe_point_colors)

        # only add the point cloud once in every 5 frames
        if self.vis_counter % 4 == 0:
            self.queue.put((pointcloud, keyframe_pose))
        
        else:
            self.queue.put((None, keyframe_pose))

    
    def display_keyframes(self):
        """ display image / depth keyframe pairs """

        if len(self.keyframe_inds) > 0:
            image_stack = []
            for keyframe_index in self.keyframe_inds:
                keyframe_image = self.images[keyframe_index]
                keyframe_depth = self.depths[keyframe_index]

                image_and_depth = vis.create_image_depth_figure(keyframe_image, keyframe_depth)
                image_stack.append(image_and_depth)

            image_stack = np.concatenate(image_stack, axis=0)
            if len(self.keyframe_inds) > 1:
                image_stack = cv2.resize(image_stack, None, fx=0.5, fy=0.5)

            cv2.imshow('keyframes', image_stack / 255.0)
            cv2.waitKey(10)

    def track(self, image):
        """ track the new frame """

        keyframe_image = self.images[self.keyframe_inds[-1]]
        images = np.stack([keyframe_image, image], axis=0)

        keyframe_pose = self.poses[self.keyframe_inds[-1]]
        poses = np.stack([keyframe_pose, self.pose_cur], axis=0)

        keyframe_depth = self.depths[self.keyframe_inds[-1]]
        depths = keyframe_depth[np.newaxis]

        edges = np.array([[0,1]], dtype=np.int32)
        fixed = np.int32(0)

        feed_dict = {
            self.images_placeholder: images,
            self.depths_placeholder: depths,
            self.poses_placeholder: poses,
            self.edges_placeholder: edges,
            self.fixed_placeholder: fixed,
            self.init_placeholder: False,
            self.intrinsics_placeholder: self.intrinsics}

        updated_poses = self.sess.run(self.outputs['poses'], feed_dict=feed_dict)

        # relative pose between keyframe and new pose
        dP = np.matmul(updated_poses[1], np.linalg.inv(updated_poses[0])) 

        # tracking probably lost, attempt recovery; sometimes caused by gaps between frames
        if pose_distance(dP) > 0.8:
            feed_dict = {
                self.images_placeholder: images,
                self.depths_placeholder: depths,
                self.poses_placeholder: poses,
                self.edges_placeholder: edges,
                self.fixed_placeholder: fixed,
                self.init_placeholder: True,
                self.intrinsics_placeholder: self.intrinsics}
                
            updated_poses = self.sess.run(self.outputs['poses'], feed_dict=feed_dict)
            dP = np.matmul(updated_poses[1], np.linalg.inv(updated_poses[0]))

        self.pose_cur = np.matmul(dP, keyframe_pose)
        return pose_distance(dP)


    def __call__(self, image, intrinsics=None):

        if intrinsics is not None:
            self.intrinsics = intrinsics

        ht, wd, _ = image.shape # get image dimensions
        did_make_new_keyframe = False

        if len(self.images) < 4: # tracking has not yet begun
            if self.index % self.rate == 0:
                self.images.append(image)
                self.depths.append(np.ones((ht, wd)))
                self.poses.append(np.eye(4))

            # initialize the tracker !
            if len(self.images) == 4:
                self.deepv2d_init()

                # set the keyframes
                self.keyframe_inds = np.random.randint(0, 4, self.n_keyframes)
                self.keyframe_inds = sorted(self.keyframe_inds.tolist())

                for i in range(3):
                    self.update_poses(fixed=False)
                    self.update_depths(fixed=False)

        else:
            dist = self.track(image)
            
            if dist > 0.8:
                new_keyframe_index = len(self.images) - 1
                query_pose = self.poses[new_keyframe_index]

                depth_new = self.reproject_depth(query_pose)
                self.depths[new_keyframe_index] = depth_new

                self.keyframe_inds.append(new_keyframe_index)
                if len(self.keyframe_inds) > self.n_keyframes:
                    old_keyframe_index = self.keyframe_inds.pop(0)
                    self.visualize_output(old_keyframe_index)

                self.update_poses(fixed=2)
                self.update_depths()


            if self.index % self.rate == 0 and (dist > 0.1):
                self.images.append(image)
                self.depths.append(np.ones((ht, wd)))
                self.poses.append(self.pose_cur)

                self.update_poses(fixed=2)
                self.update_depths()

            # make a new keyfrane
            if len(self.images) - self.keyframe_inds[-1] >= self.window:
                new_keyframe_index = self.keyframe_inds[-1] + 2
                query_pose = self.poses[new_keyframe_index]

                depth_new = self.reproject_depth(query_pose)
                self.depths[new_keyframe_index] = depth_new

                self.keyframe_inds.append(new_keyframe_index)
                if len(self.keyframe_inds) > self.n_keyframes:
                    old_keyframe_index = self.keyframe_inds.pop(0)
                    self.visualize_output(old_keyframe_index)

                self.update_poses(fixed=2)
                self.update_depths()


        self.display_keyframes()
        self.index += 1


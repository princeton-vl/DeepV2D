import tensorflow as tf
import numpy as np
from data_stream import kitti_utils
from data_stream import util

import csv
import cv2
import os
import time
import random
import glob


class KittiRaw(object):
    default_args = {
        'frames': 4,
        'radius': 2,
        'height': 300,
        'width': 1088,
        'crop': 108,
        'scale': 0.1,
    }

    def __init__(self, dataset_path, mode='train', args=default_args):

        self.dataset_path = dataset_path
        self.args = args
        self.mode = mode

        self._collect_scenes()
        self._build_training_set_index()


    def __len__(self):
        return len(self.training_set_index)

    def __getitem__(self, index):
        return self._load_example(self.training_set_index[index])

    def _load_example(self, sequence):

        n_frames = len(sequence)
        scene = sequence[0]['drive']

        center_idx = 2
        # put the keyframe at the first index
        sequence = [sequence[center_idx]] + \
            [sequence[i] for i in range(n_frames) if not i==center_idx]

        images, poses = [], []
        for frame in sequence:
            img = self._load_image(frame['image'])
            images.append(img)
            poses.append(frame['pose'])

        depth = self._load_depth(sequence[0]['velo'], images[0], scene)
        filled = util.fill_depth(depth)

        intrinsics = self._load_intrinsics(img, scene).astype("float32")
        for i in range(len(images)):
            images[i] = cv2.resize(images[i], (self.args['width'], self.args['height']))
            images[i] = images[i][self.args['crop']:]

        images = np.array(images, dtype="float32")
        poses = np.array(poses, dtype="float32")
        depth = depth.astype("float32")
        filled = filled.astype("float32")

        example_blob = {
            'images': images,
            'depth': depth,
            'filled': filled,
            'poses': poses,
            'pred': filled,
            'intrinsics': intrinsics,
        }

        return example_blob


    def _fetch_image_path(self, drive, index):
        image_path = os.path.join(drive[:10], drive+'_sync', 'image_02', 'data', '%010d.png'%index)
        return os.path.join(self.dataset_path, image_path)

    def _fetch_velo_path(self, drive, index):
        velo_path = os.path.join(drive[:10], drive+'_sync', 'velodyne_points', 'data', '%010d.bin'%index)
        return os.path.join(self.dataset_path, velo_path)

    def test_set_iterator(self, radius=2):
        test_list = []
        with open('data/kitti/test_files_eigen.txt') as f:
            reader = csv.reader(f)
            for row in reader:
                test_list.append(row[0])

        self.poses = {}
        self.calib = {}
        for test_frame in test_list:
            comps = test_frame.split('/')
            drive = comps[1].replace('_sync', '')
            frame = int(comps[4].replace('.png', ''))

            if drive not in self.poses:
                trajectory = self._read_oxts_data(drive)
                proj_c2p, proj_v2c, imu2cam = self._read_raw_calib_data(drive)

                for i in range(len(trajectory)):
                    trajectory[i] = np.dot(imu2cam, util.inv_SE3(trajectory[i]))
                    trajectory[i][0:3, 3] *= self.args['scale']

                self.poses[drive] = trajectory
                self.calib[drive] = (proj_c2p, proj_v2c, imu2cam)

            else:
                trajectory = self.poses[drive]

            seq = []
            for j in range(frame-radius, frame+radius+1):
                j = min(max(0, j), len(trajectory)-1)
                frame = {
                    'image': self._fetch_image_path(drive, j),
                    'velo': self._fetch_velo_path(drive, j),
                    'pose': self.poses[drive][j],
                    'drive': drive,
                }
                seq.append(frame)

            data_blob = self._load_example(seq)
            yield data_blob['images'], data_blob['intrinsics']


    def test_drive(self, drive, radius=2):

        self.poses = {}
        self.calib = {}
        trajectory = self._read_oxts_data(drive)
        proj_c2p, proj_v2c, imu2cam = self._read_raw_calib_data(drive)

        for i in range(len(trajectory)):
            trajectory[i] = np.dot(imu2cam, util.inv_SE3(trajectory[i]))
            trajectory[i][0:3, 3] *= self.args['scale']

        self.poses[drive] = trajectory
        self.calib[drive] = (proj_c2p, proj_v2c, imu2cam)

        for i in range(len(trajectory)):
            seq = []
            for j in range(i-radius, i+radius+1):
                j = min(max(0, j), len(trajectory)-1)
                frame = {
                    'image': self._fetch_image_path(drive, j),
                    'velo': self._fetch_velo_path(drive, j),
                    'pose': self.poses[drive][j],
                    'drive': drive,
                }
                seq.append(frame)

            yield self._load_example(seq)

    def iterate_sequence(self, drive):
        trajectory = self._read_oxts_data(drive)

        for i in range(len(trajectory)):
            imfile = self._fetch_image_path(drive, i)
            image = self._load_image(imfile)

            image = cv2.resize(image, (self.args['width'], self.args['height']))
            image = image[self.args['crop']:]

            proj_c2p, proj_v2c, imu2cam = self._read_raw_calib_data(drive)
            proj_c2p[0] *= self.args['width'] / float(image.shape[1])
            proj_c2p[1] *= self.args['height'] / float(image.shape[0])

            fx = proj_c2p[0, 0]
            fy = proj_c2p[1, 1]
            cx = proj_c2p[0, 2]
            cy = proj_c2p[1, 2] - self.args['crop']

            intrinsics = np.array([fx, fy, cx, cy])
            yield image, intrinsics
    

    def _build_training_set_index(self, radius=2):
        self.training_set_index = []
        self.poses = {}
        self.calib = {}

        for drive in self.sequences:

            trajectory = self._read_oxts_data(drive)
            proj_c2p, proj_v2c, imu2cam = self._read_raw_calib_data(drive)

            for i in range(len(trajectory)):
                trajectory[i] = np.dot(imu2cam, util.inv_SE3(trajectory[i]))
                trajectory[i][0:3, 3] *= self.args['scale']

            self.poses[drive] = trajectory
            self.calib[drive] = (proj_c2p, proj_v2c, imu2cam)

            for i in range(len(trajectory)):
                seq = []
                for j in range(i-radius, i+radius+1):
                    j = min(max(0, j), len(trajectory)-1)
                    frame = {
                        'image': self._fetch_image_path(drive, j),
                        'velo': self._fetch_velo_path(drive, j),
                        'pose': self.poses[drive][j],
                        'drive': drive,
                    }
                    seq.append(frame)
                self.training_set_index.append(seq)


    def _load_intrinsics(self, img, drive):
        proj_c2p, proj_v2c, imu2cam = self.calib[drive]
        proj_c2p = proj_c2p.copy()
        proj_c2p[0] *= self.args['width'] / float(img.shape[1])
        proj_c2p[1] *= self.args['height'] / float(img.shape[0])

        fx = proj_c2p[0, 0]
        fy = proj_c2p[1, 1]
        cx = proj_c2p[0, 2]
        cy = proj_c2p[1, 2] - self.args['crop']

        intrinsics = np.array([fx, fy, cx, cy])
        return intrinsics


    def _load_image(self, image_path):
        return cv2.imread(image_path)


    def _load_depth(self, velo_path, img, drive):
        points = np.fromfile(velo_path, dtype=np.float32).reshape(-1, 4)
        points[:, 3] = 1.0  # homogeneous
        proj_c2p, proj_v2c, imu2cam = self.calib[drive]

        proj_c2p = proj_c2p.copy()
        proj_c2p[0] *= self.args['width'] / float(img.shape[1])
        proj_c2p[1] *= self.args['height'] / float(img.shape[0])

        sz = [self.args['height'], self.args['width']]
        depth = kitti_utils.velodyne_to_depthmap(points, sz, proj_c2p, proj_v2c)
        depth = depth[self.args['crop']:]
        return depth * self.args['scale']


    def _collect_scenes(self):
        if self.mode == 'train':
            sequence_list = 'data/kitti/train_scenes_eigen.txt'
        if self.mode == 'test':
            sequence_list = 'data/kitti/test_scenes_eigen.txt'

        with open(sequence_list) as f:
            reader = csv.reader(f)
            sequences = [x[0] for x in reader]

        self.sequences = sequences


    def _read_oxts_data(self, drive):
        oxts_path = os.path.join(self.dataset_path,
            drive[:10], drive+'_sync', 'oxts', 'data', '*.txt')
        oxts_files = sorted(glob.glob(oxts_path))
        trajectory = []
        for x in kitti_utils.get_oxts_packets_and_poses(oxts_files):
            trajectory.append(x.T_w_imu)

        return trajectory


    def _read_raw_calib_file(self, filepath):
        # From https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
        """Read in a calibration file and parse into a dictionary."""

        data = {}
        with open(filepath, 'r') as f:
            for line in f.readlines():
                key, value = line.split(':', 1)
                # The only non-float values in these files are dates, which
                # we don't care about anyway
                try:
                        data[key] = np.array([float(x) for x in value.split()])
                except ValueError:
                        pass
        return data


    def _read_raw_calib_data(self, drive, cam=2):
        # From https://github.com/mrharicot/monodepth/blob/master/utils/evaluation_utils.py

        drive = drive + '_sync'
        cam_to_cam_filepath = os.path.join(self.dataset_path, drive[:10], 'calib_cam_to_cam.txt')
        imu_to_velo_filepath = os.path.join(self.dataset_path, drive[:10], 'calib_imu_to_velo.txt')
        velo_to_cam_filepath = os.path.join(self.dataset_path, drive[:10], 'calib_velo_to_cam.txt')

        cam2cam = self._read_raw_calib_file(cam_to_cam_filepath)
        velo2cam = self._read_raw_calib_file(velo_to_cam_filepath)
        imu2velo = self._read_raw_calib_file(imu_to_velo_filepath)

        imu2velo = np.hstack((imu2velo['R'].reshape(3,3), imu2velo['T'][..., np.newaxis]))
        imu2velo = np.vstack((imu2velo, np.array([0, 0, 0, 1.0])))

        velo2cam = np.hstack((velo2cam['R'].reshape(3,3), velo2cam['T'][..., np.newaxis]))
        velo2cam = np.vstack((velo2cam, np.array([0, 0, 0, 1.0])))

        R_cam2rect = np.eye(4)
        R_cam2rect[:3,:3] = cam2cam['R_rect_00'].reshape(3,3)
        P_rect = cam2cam['P_rect_0'+str(cam)].reshape(3,4)

        proj_c2p = np.dot(P_rect, R_cam2rect)
        proj_v2c = velo2cam

        imu2cam = np.dot(velo2cam, imu2velo)
        return proj_c2p, proj_v2c, imu2cam

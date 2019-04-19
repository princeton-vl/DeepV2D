import tensorflow as tf
import numpy as np
from data_stream import util

import csv
import cv2
import os
import time
import random
import glob

class NYU(object):
    default_args = {
        'frames': 3,
        'radius': 4,
        'tdelta': .25,
    }
    def __init__(self, dataset_path, mode='test', tmp_path='tmp', args=default_args):
        if not os.path.exists(tmp_path):
            os.makedirs(tmp_path)
        self.dataset_path = dataset_path
        self.tmp_path = tmp_path
        self.args = args
        self.mode = mode

        self._build_sequence_set_index()

    def __len__(self):
        return len(self.sequence_set_index)

    def __getitem__(self, index):
        return self._load_example(self.sequence_set_index[index])

    def get_dims(self):
        return (self.args['frames']+1, 480, 640)

    def set_depth(self, index, val):
        self._write_depth(index, val)

    def _get_intrinsics(self):
        fx = 5.1885790117450188e+02
        fy = 5.1946961112127485e+02
        cx = 3.2558244941119034e+02
        cy = 2.5373616633400465e+02

        intrinsics = np.array([fx, fy, cx, cy], dtype="float32")
        return intrinsics


    def _load_test(self, sequence):

        n_frames = len(sequence)
        idxs = [i for i in range(n_frames) if not i==(n_frames/2)]
        idxs = np.random.choice(idxs, size=self.args.get('frames'), replace=False)
        sequence = [sequence[n_frames/2]] + [sequence[i] for i in idxs]

        images, poses = [], []
        for frame in sequence:
            image = self._undistort(cv2.imread(frame['image']))
            images.append(image)

        images = np.array(images, dtype="float32")
        intrinsics = self._get_intrinsics()

        example_blob = {
            'images': images,
            'intrinsics': intrinsics,
        }

        return example_blob


    def _load_example(self, sequence, sample=False):

        n_frames = len(sequence)

        idxs = [i for i in range(n_frames) if not i==(n_frames/2)]
        if sample:
            idxs = np.random.choice(idxs, size=self.args.get('frames'), replace=False)

        sequence = [sequence[n_frames/2]] + [sequence[i] for i in idxs]

        images, poses = [], []
        for frame in sequence:
            image = self._undistort(cv2.imread(frame['image']))
            images.append(image)
            poses.append(frame['pose'])

        images = np.array(images, dtype="uint8")
        poses = np.array(poses, dtype="float32")

        depth = cv2.imread(sequence[0]['depth'], cv2.IMREAD_ANYDEPTH)
        depth = (depth / 5000.0).astype("float32")

        filled = util.fill_depth(depth).astype("float32")
        intrinsics = self._get_intrinsics()

        example_blob = {
            'images': images,
            'depth': depth,
            'filled': filled,
            'poses': poses,
            'intrinsics': intrinsics,
        }

        return example_blob


    def _undistort(self, image):
        k1 =  2.0796615318809061e-01
        k2 = -5.8613825163911781e-01
        p1 = 7.2231363135888329e-04
        p2 = 1.0479627195765181e-03
        k3 = 4.9856986684705107e-01

        fx = 5.1885790117450188e+02
        fy = 5.1946961112127485e+02
        cx = 3.2558244941119034e+02
        cy = 2.5373616633400465e+02

        kmat = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        dist = np.array([[k1, k2, p1, p2, k3]])
        image = cv2.undistort(image, kmat, dist)
        return image


    def _write_depth(self, idx, depth):
        depth_path = os.path.join(self.tmp_path, '%06d.png'%idx)
        depth16 = (depth*5000.0).astype("uint16")
        cv2.imwrite(depth_path, depth16)

    def _load_pred(self, idx):
        depth_path = os.path.join(self.tmp_path, '%06d.png'%idx)
        if not os.path.isfile(depth_path):
            return None
        depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)/5000.0
        return depth.astype("float32")


    def _load_filled(self, idx, depth):
        filled_path = os.path.join(self.tmp_path, '%06d_filled.png'%idx)
        if not os.path.isfile(filled_path):
            filled = util.fill_depth(depth).astype("float32")
            cv2.imwrite(filled_path, (filled*5000).astype("uint16"))
        else:
            filled = cv2.imread(filled_path, cv2.IMREAD_ANYDEPTH)/5000.0

        return filled.astype("float32")


    def _load_trajectory_data(self, scene):
        scene_dir = os.path.join(self.dataset_path, scene)
        associations_file = os.path.join(scene_dir, 'associations.txt')
        camera_file = os.path.join(scene_dir, 'pose.txt')

        pairs = {}
        with open(associations_file) as f:
            reader = csv.reader(f, delimiter=' ')
            for row in reader:
                pairs[row[0]] = (row[1], row[3])

        poses = {}
        timestamps = []
        with open(camera_file) as f:
            reader = csv.reader(f, delimiter=' ')
            for row in reader:
                tstamp, vec = row[0], [float(x) for x in row[1:]]
                timestamps.append(tstamp)
                vec = np.array(vec).reshape([1, 7])
                pose = util.pose_vec2mat(vec)
                poses[tstamp] = pose[0]

        return timestamps, pairs, poses


    def _make_video(self, i0, timestamps, radius=3):

        idxs_forward = []
        idxs_reverse = []
        t0 = timestamps[i0]

        dt = self.args.get('tdelta')
        for i in range(radius):
            i_forward = np.argmin(np.abs((t0+(i+1)*dt)-timestamps))
            i_reverse = np.argmin(np.abs((t0-(i+1)*dt)-timestamps))

            idxs_forward.append(i_forward)
            idxs_reverse.append(i_reverse)

        idxs = idxs_reverse[::-1] + [i0] + idxs_forward
        return idxs


    def _build_sequence_set_index(self, tskip=0.6):
        self.sequence_set_index = []
        for sceneb in self._collect_train_scenes(mode='train'):
            seach_str = os.path.join(self.dataset_path, '%s*'%sceneb)
            for scene in glob.glob(seach_str):
                scene = os.path.basename(scene)
                try:
                    timestamps, pairs, poses = self._load_trajectory_data(scene)
                except:
                    continue

                tstamps = np.array(timestamps, dtype="float64")
                tcur = tstamps[0] - 2*tskip

                for i in range(0, len(timestamps)):
                    if tstamps[i] - tcur > tskip:
                        tcur = tstamps[i]
                        vid = self._make_video(i, tstamps, radius=self.args.get('radius'))

                        sequence = []
                        for j in range(len(vid)):
                            t = timestamps[vid[j]]
                            frame = {
                                'image': os.path.join(self.dataset_path, scene, pairs[t][0]),
                                'depth': os.path.join(self.dataset_path, scene, pairs[t][1]),
                                'pose': util.inv_SE3(poses[t]),
                            }
                            sequence.append(frame)

                        self.sequence_set_index.append(sequence)


    def _load_scene(self, scene):
        image_list = os.path.join(self.dataset_path, scene, 'rgb.txt')
        timestamps, images = [], []
        with open(image_list) as f:
            reader = csv.reader(f, delimiter=' ')
            for row in reader:
                timestamps.append(row[0])
                images.append(row[1])
        return timestamps, images


    def test_set_iterator(self):
        with open('data/nyu/eigen_test_files.txt') as f:
            test_list = [x[0] for x in csv.reader(f)]

        for test_frame in test_list:
            scene = test_frame.split('/')[0]
            query = os.path.join('rgb', test_frame.split('-')[1] + '.png')

            timestamps, imfiles = self._load_scene(scene)
            tstamps = np.array(timestamps, dtype="float64")
            for i, t in enumerate(timestamps):
                if imfiles[i] == query:
                    vid = self._make_video(i, tstamps, radius=4)
                    seq = []
                    for j in range(len(vid)):
                        t = timestamps[vid[j]]
                        frame = {
                            'image': os.path.join(self.dataset_path, scene, imfiles[vid[j]]),
                        }
                        seq.append(frame)
                    yield self._load_test(seq)



    def _collect_train_scenes(self, mode='train'):
        scenes_list = 'data/nyu/train_scenes.txt'
        reader = csv.reader(open(scenes_list))
        scenes = [x[0] for x in reader]

        return scenes

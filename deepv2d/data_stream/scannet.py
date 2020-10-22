import numpy as np
import os
import cv2
import re
import csv
import glob
import random
import pickle


from scipy import interpolate

def fill_depth(depth):
    x, y = np.meshgrid(np.arange(depth.shape[1]).astype("float32"),
                       np.arange(depth.shape[0]).astype("float32"))
    xx = x[depth > 0]
    yy = y[depth > 0]
    zz = depth[depth > 0]

    grid = interpolate.griddata((xx, yy), zz.ravel(),
                                (x, y), method='nearest')
    return grid

class ScanNet:
    def __init__(self, dataset_path, n_frames=4, r=6):
        self.dataset_path = dataset_path
        self.n_frames = n_frames
        self.height = 480
        self.width = 640
        self.build_dataset_index(r=r)

    def __len__(self):
        return len(self.dataset_index)

    def shape(self):
        return [self.n_frames, self.height, self.width]

    def __getitem__(self, index):
        data_blob = self.dataset_index[index]
        num_frames = data_blob['n_frames']
        num_samples = self.n_frames - 1

        frameid = data_blob['id']
        keyframe_index = num_frames // 2

        inds = np.arange(num_frames)
        inds = inds[~np.equal(inds, keyframe_index)]
        
        inds = np.random.choice(inds, num_samples, replace=False)
        inds = [keyframe_index] + inds.tolist()

        images = []
        for i in inds:
            image = cv2.imread(data_blob['images'][i])
            image = cv2.resize(image, (640, 480))
            images.append(image)

        poses = []
        for i in inds:
            poses.append(data_blob['poses'][i])

        images = np.stack(images, axis=0).astype(np.uint8)
        poses = np.stack(poses, axis=0).astype(np.float32)

        depth_file = data_blob['depth']
        depth = cv2.imread(depth_file, cv2.IMREAD_ANYDEPTH)
        
        depth = (depth.astype(np.float32)) / 1000.0
        filled = fill_depth(depth)
        
        K = data_blob['intrinsics']
        kvec = np.stack([K[0,0], K[1,1], K[0,2], K[1,2]], axis=0)

        depth = depth[...,None]
        return images, poses, depth, filled, filled, kvec, frameid


    def __iter__(self):
        random.shuffle(self.dataset_index)
        while 1:
            self.perm = np.arange(len(self.dataset_index))
            np.random.shuffle(self.perm)
            for i in self.perm:
                yield self.__getitem__(i)

    def _load_scan(self, scan):
        scan_path = os.path.join(self.dataset_path, scan)
        datum_file = os.path.join(scan_path, 'pickle-scene.pkl')

        if not os.path.isfile(datum_file):

            imfiles = glob.glob(os.path.join(scan_path, 'pose', '*.txt'))
            ixs = sorted([int(os.path.basename(x).split('.')[0]) for x in imfiles])

            images = []
            for i in ixs[::2]:
                imfile = os.path.join(scan_path, 'color', '%d.jpg'%i)
                images.append(imfile)

            poses = []
            for i in ixs[::2]:
                posefile = os.path.join(scan_path, 'pose', '%d.txt' % i)
                pose = np.loadtxt(posefile, delimiter=' ').astype(np.float32)  
                poses.append(np.linalg.inv(pose)) # convert c2w->w2c

            depths = []
            for i in ixs[::2]:
                depthfile = os.path.join(scan_path, 'depth', '%d.png'%i)
                depths.append(depthfile)

            color_intrinsics = np.loadtxt(os.path.join(scan_path, 'intrinsic', 'intrinsic_color.txt'), delimiter=' ')
            depth_intrinsics = np.loadtxt(os.path.join(scan_path, 'intrinsic', 'intrinsic_depth.txt'), delimiter=' ')

            datum = images, depths, poses, color_intrinsics, depth_intrinsics
            pickle.dump(datum, open(datum_file, 'wb')) 
            
        else:
            datum = pickle.load(open(datum_file, 'rb'))

        return datum


    def build_dataset_index(self, r=4, skip=12):
        self.dataset_index = []
        data_id = 0

        for scan in sorted(os.listdir(self.dataset_path)):

            scanid = int(re.findall(r'scene(.+?)_', scan)[0])
            if scanid>660:
                continue

            images, depths, poses, color_intrinsics, depth_intrinsics = self._load_scan(scan)

            for i in range(r, len(images)-r, skip):
                # some poses in scannet are nans
                if np.any(np.isnan(poses[i-r:i+r+1])):
                    continue

                training_example = {}
                training_example['depth'] = depths[i]
                training_example['images'] = images[i-r:i+r+1]
                training_example['poses'] = poses[i-r:i+r+1]
                training_example['intrinsics'] = depth_intrinsics
                training_example['n_frames'] = 2*r+1
                training_example['id'] = data_id

                self.dataset_index.append(training_example)
                data_id += 1

    def test_set_iterator(self):

        test_frames = np.loadtxt('data/scannet/scannet_test.txt', dtype=np.unicode_)
        test_data = []

        for i in range(0, len(test_frames), 4):
            test_frame_1 = str(test_frames[i]).split('/')
            test_frame_2 = str(test_frames[i+1]).split('/')
            scan = test_frame_1[3]

            imageid_1 = int(re.findall(r'frame-(.+?).color.jpg', test_frame_1[-1])[0])
            imageid_2 = int(re.findall(r'frame-(.+?).color.jpg', test_frame_2[-1])[0])            
            test_data.append((scan, imageid_1, imageid_2))

        # random.shuffle(test_data)        
        for (scanid, imageid_1, imageid_2) in test_data:

            scandir = os.path.join(self.dataset_path, scanid)
            num_frames = len(os.listdir(os.path.join(scandir, 'color')))

            images = []

            # we need to include imageid_2 and imageid_1 to compare to BA-Net poses,
            # then sample remaining 6 frames uniformly
            dt = imageid_2 - imageid_1
            s = 3

            for i in [0, dt, -3*s, -2*s, -s, s, 2*s, 3*s]:
                otherid = min(max(1, i+imageid_1), num_frames-1)
                image_file = os.path.join(scandir, 'color', '%d.jpg'%otherid)
                image = cv2.imread(image_file)
                image = cv2.resize(image, (640, 480))
                images.append(image)

            depth_file = os.path.join(scandir, 'depth', '%d.png'%imageid_1)
            depth = cv2.imread(depth_file, cv2.IMREAD_ANYDEPTH)
            depth = (depth/1000.0).astype(np.float32)

            pose1 = np.loadtxt(os.path.join(scandir, 'pose', '%d.txt'%imageid_1), delimiter=' ')
            pose2 = np.loadtxt(os.path.join(scandir, 'pose', '%d.txt'%imageid_2), delimiter=' ')
            pose1 = np.linalg.inv(pose1)
            pose2 = np.linalg.inv(pose2)
            pose_gt = np.dot(pose2, np.linalg.inv(pose1))

            depth_intrinsics = os.path.join(scandir, 'intrinsic/intrinsic_depth.txt')
            K = np.loadtxt(depth_intrinsics, delimiter=' ')
            fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]

            images = np.stack(images, axis=0).astype(np.uint8)
            depth = depth.astype(np.float32)
            intrinsics = np.array([fx, fy, cx, cy], dtype=np.float32)

            data_blob = {
                'images': images,
                'depth': depth,
                'pose': pose_gt,
                'intrinsics': intrinsics,
            }

            yield data_blob


    def iterate_sequence(self, scan):
        scan_path = os.path.join(self.dataset_path, scan)
        imfiles = glob.glob(os.path.join(scan_path, 'pose', '*.txt'))
        ixs = sorted([int(os.path.basename(x).split('.')[0]) for x in imfiles])

        K = np.loadtxt(os.path.join(scan_path, 'intrinsic', 'intrinsic_depth.txt'), delimiter=' ')
        fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
        intrinsics = np.array([fx, fy, cx, cy], dtype=np.float32)

        images = []
        for i in ixs:
            imfile = os.path.join(scan_path, 'color', '%d.jpg'%i)
            image = cv2.imread(imfile)
            image = cv2.resize(image, (640, 480))
            yield image, intrinsics
import sys
sys.path.append('lib')

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import cv2
import os
import time
import argparse
import glob

from core import config
from deepv2d import DeepV2DSLAM


INPUT_DIMS = [7, 480, 640]

def load_slam_sequence(path, n_frames=7):
    images = []
    for imfile in sorted(glob.glob(os.path.join(path, "*.png"))):
        images.append(cv2.imread(imfile))

    poses = []
    for posefile in sorted(glob.glob(os.path.join(path, "pose_*.txt"))):
        poses.append(np.loadtxt(posefile))

    poses = np.array(poses, dtype=np.float32)
    intrinsics = np.loadtxt(os.path.join(path, 'intrinsics.txt'))

    return images, poses, intrinsics


def plot_trajectory(ax, poses, name):

    poses = np.array(poses)
    for i in range(poses.shape[0]):
        poses[i] = np.dot(poses[i], np.linalg.inv(poses[0]))

    traj = np.linalg.inv(poses)[:, :3, 3]   
    X, Y, Z = traj[:,0], traj[:,1], traj[:,2]
    line,  = ax.plot(X, Y, Z, label=name)
    
    x0, x1 = X.min(), X.max()
    y0, y1 = Y.min(), Y.max()
    z0, z1 = Z.min(), Z.max()

    upper = np.min([x0, y0, z0]) - 0.2
    lower = np.max([x1, y1, z1]) + 0.2

    sx = x1 - x0
    sy = y1 - y0
    sz = z1 - z0

    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_zlim(lower, upper)

    return line


def main(args):
    cfg = config.cfg_from_file(args.cfg)
    net = DeepV2DSLAM(INPUT_DIMS, cfg)

    images, orb_poses, intrinsics = load_slam_sequence(args.sequence)
    fig1 = plt.figure()
    fig2, (ax1, ax2) = plt.subplots(1,2)
    ax = fig1.add_subplot(111, projection='3d')

    with tf.Session() as sess:
        net.restore(sess, args.model)
        net.set_fcrn_weights(sess)
        net.set_intrinsics(intrinsics)

        for image in images:
            ##### Update the tracker #####
            start = time.time()
            net.update(image)
            stop = time.time()
            print("Iteration Time: %f" % (stop-start))

             ##### Display the results #####
            ax.cla()
            ax1.cla()
            ax2.cla()
            ax1.imshow(net.keyframe_image[...,::-1]/255.0)
            ax2.imshow(net.keyframe_depth[...,0])
            plot_trajectory(ax, net.poses, name='DeepV2D')
            plot_trajectory(ax, orb_poses[:len(net.poses)+1], name='RGB-D ORB-SLAM')
            ax.legend()
            plt.pause(0.05)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='cfgs/nyu.yaml', help='config file used to train the model')
    parser.add_argument('--model', default='models/nyu/_stage_2.ckpt-120000', help='path to model checkpoint')
    parser.add_argument('--sequence', help='path to kitti sequence folder')
    args = parser.parse_args()

    main(args)

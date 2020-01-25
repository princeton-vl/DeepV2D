import numpy as np
import csv
import os
import time
import random
import glob
import os.path as osp


from geometry.transformation import *
import cv2
import tensorflow as tf


fx = 517.3
fy = 516.5
cx = 318.6
cy = 255.3
intrinsics = np.array([fx, fy, cx, cy])

def associate_frames(image_times, depth_times, pose_times):
    associations = []
    for i, t in enumerate(image_times):
        j = np.argmin(np.abs(depth_times - t))
        k = np.argmin(np.abs(pose_times - t))
        associations.append((i, j, k))
    return associations


import argparse
import random
import numpy
import sys

_EPS = numpy.finfo(float).eps * 4.0

def transform44(l):
    """
    Generate a 4x4 homogeneous transformation matrix from a 3D point and unit quaternion.
    
    Input:
    l -- tuple consisting of (stamp,tx,ty,tz,qx,qy,qz,qw) where
         (tx,ty,tz) is the 3D position and (qx,qy,qz,qw) is the unit quaternion.
         
    Output:
    matrix -- 4x4 homogeneous transformation matrix
    """
    t = l[1:4]
    q = numpy.array(l[4:8], dtype=numpy.float64, copy=True)
    nq = numpy.dot(q, q)
    if nq < _EPS:
        return numpy.array((
        (                1.0,                 0.0,                 0.0, t[0])
        (                0.0,                 1.0,                 0.0, t[1])
        (                0.0,                 0.0,                 1.0, t[2])
        (                0.0,                 0.0,                 0.0, 1.0)
        ), dtype=numpy.float64)
    q *= numpy.sqrt(2.0 / nq)
    q = numpy.outer(q, q)
    return numpy.array((
        (1.0-q[1, 1]-q[2, 2],     q[0, 1]-q[2, 3],     q[0, 2]+q[1, 3], t[0]),
        (    q[0, 1]+q[2, 3], 1.0-q[0, 0]-q[2, 2],     q[1, 2]-q[0, 3], t[1]),
        (    q[0, 2]-q[1, 3],     q[1, 2]+q[0, 3], 1.0-q[0, 0]-q[1, 1], t[2]),
        (                0.0,                 0.0,                 0.0, 1.0)
        ), dtype=numpy.float64)


class TUM_RGBD:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path

    def iterate_sequence(self, sequence_name, matrix=False):
        """returns list of images, depths, and poses"""
        sequence_dir = os.path.join(self.dataset_path, sequence_name)
        image_list = os.path.join(sequence_dir, 'rgb.txt')
        depth_list = os.path.join(sequence_dir, 'depth.txt')
        pose_list = os.path.join(sequence_dir, 'groundtruth.txt')

        image_data = np.loadtxt(image_list, delimiter=' ', dtype=np.unicode_, skiprows=3)
        depth_data = np.loadtxt(depth_list, delimiter=' ', dtype=np.unicode_, skiprows=3)

        try:
            pose_data = np.loadtxt(pose_list, delimiter=' ', dtype=np.float64, skiprows=3)
        except:
            pose_data = np.zeros((len(image_data), 7))
            secret = True

        intrinsics_mat = intrinsics.copy()

        images = []
        for (tstamp, image_file) in image_data:
            image_file = os.path.join(sequence_dir, image_file)
            image = cv2.imread(image_file)

            yield image, intrinsics_mat

        #     images.append(image)

        # depths = []
        # for (_, depth_file) in depth_data:
        #     depth_file = os.path.join(sequence_dir, depth_file)
        #     depth = cv2.imread(depth_file, cv2.IMREAD_ANYDEPTH)
        #     depth = depth.astype(np.float32) / 5000.0
        #     depths.append(depth)

        # traj_gt = []
        # for pose_vec in pose_data:
        #     if matrix:
        #         traj_gt.append(transform44(pose_vec))
        #     else:
        #         traj_gt.append(pose_vec)

        # image_times = image_data[:,0].astype(np.float64)
        # depth_times = depth_data[:,0].astype(np.float64)
        # pose_times = pose_data[:,0].astype(np.float64)
        # indicies = associate_frames(image_times, depth_times, pose_times)

        # rgbd_images = []
        # rgbd_depths = []
        # timestamps = []
        # for (img_ix, depth_ix, pose_ix) in indicies:
        #     timestamps.append(image_times[img_ix])
        #     rgbd_images.append(images[img_ix])
        #     rgbd_depths.append(depths[depth_ix])
            
        # timestamps = np.stack(timestamps, axis=0)
        # rgbd_images = np.stack(rgbd_images, axis=0)
        # rgbd_depths = np.stack(rgbd_depths, axis=0)
        # intrinsics_mat = intrinsics.copy()

        # for img in rgbd_images:
        #     yield img, intrinsics_mat
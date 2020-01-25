import sys
sys.path.append('deepv2d')

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import cv2
import os
import time
import argparse
import glob

import vis
from core import config
from data_stream.scannet import ScanNet
from deepv2d import DeepV2D

import eval_utils


def write_to_folder(images, intrinsics, test_id):
    dest = os.path.join("scannet/%06d" % test_id)

    if not os.path.isdir(dest):
        os.makedirs(dest)

    for i, img in enumerate(images):
        cv2.imwrite(os.path.join(dest, '%d.png'%i), img)

    np.savetxt(os.path.join(dest, 'intrinsics.txt'), intrinsics)



def make_predictions(args):

    cfg = config.cfg_from_file(args.cfg)
    deepv2d = DeepV2D(cfg, args.model, use_fcrn=True, mode=args.mode)

    with tf.Session() as sess:
        deepv2d.set_session(sess)

        depth_predictions, pose_predictions = [], []
        depth_groundtruth, pose_groundtruth = [], []
        db = ScanNet(args.dataset_dir)

        for test_id, test_blob in enumerate(db.test_set_iterator()):
            images, intrinsics = test_blob['images'], test_blob['intrinsics']
            depth_pred, poses_pred = deepv2d(images, intrinsics)

            # use keyframe depth for evaluation
            depth_predictions.append(depth_pred[0])
            
            # BA-Net evaluates pose as the relative transformation between two frames
            delta_pose = poses_pred[1] @ np.linalg.inv(poses_pred[0])
            pose_predictions.append(delta_pose)

            depth_groundtruth.append(test_blob['depth'])
            pose_groundtruth.append(test_blob['pose'])


    predictions = (depth_predictions, pose_predictions)
    groundtruth = (depth_groundtruth, pose_groundtruth)
    return groundtruth, predictions


def evaluate(groundtruth, predictions):
    pose_results = {}
    depth_results = {}

    depth_groundtruth, pose_groundtruth = groundtruth
    depth_predictions, pose_predictions = predictions
    
    num_test = len(depth_groundtruth)
    for i in range(num_test):
        # match scales using median
        scalor = eval_utils.compute_scaling_factor(depth_groundtruth[i], depth_predictions[i])
        depth_predictions[i] = scalor * depth_predictions[i]

        depth_metrics = eval_utils.compute_depth_errors(depth_groundtruth[i], depth_predictions[i])
        pose_metrics = eval_utils.compute_pose_errors(pose_groundtruth[i], pose_predictions[i])

        if i == 0:
            for pkey in pose_metrics:
                pose_results[pkey] = []
            for dkey in depth_metrics:
                depth_results[dkey] = []

        for pkey in pose_metrics:
            pose_results[pkey].append(pose_metrics[pkey])

        for dkey in depth_metrics:
            depth_results[dkey].append(depth_metrics[dkey])


    ### aggregate metrics
    for pkey in pose_results:
        pose_results[pkey] = np.mean(pose_results[pkey])

    for dkey in depth_results:
        depth_results[dkey] = np.mean(depth_results[dkey])

    print(("{:>1}, "*len(depth_results)).format(*depth_results.keys()))
    print(("{:10.4f}, "*len(depth_results)).format(*depth_results.values()))

    print(("{:>16}, "*len(pose_results)).format(*pose_results.keys()))
    print(("{:16.4f}, "*len(pose_results)).format(*pose_results.values()))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='cfgs/scannet.yaml', help='config file used to train the model')
    parser.add_argument('--model', default='models/scannet.ckpt', help='path to model checkpoint')
    parser.add_argument('--dataset_dir', help='path to scannet dataset')


    parser.add_argument('--mode', default='keyframe', help='config file used to train the model')
    parser.add_argument('--fcrn', action="store_true", help='use single image depth initializiation')
    parser.add_argument('--n_iters', type=int, default=8, help='number of video frames to use for reconstruction')
    parser.add_argument('--viz', action="store_true", help='display depth maps during inference')

    args = parser.parse_args()

    groundtruth, predictions = make_predictions(args)
    evaluate(groundtruth, predictions)
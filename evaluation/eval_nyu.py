import sys
sys.path.append('deepv2d')

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tqdm import tqdm
import cv2
import os
import time
import argparse
import glob

import vis
from core import config
from deepv2d import DeepV2D
import eval_utils



def load_test_sequence(path, n_frames=-1):
    """ loads images and intrinsics from demo folder """
    images = []
    for imfile in sorted(glob.glob(os.path.join(path, "*.png"))):
        img = cv2.imread(imfile)
        images.append(img)

    inds = np.arange(1, len(images))
    if n_frames > 0:
        inds = np.random.choice(inds, n_frames, replace=False)

    inds = [0] + inds.tolist() # put keyframe image first
    images = [images[i] for i in inds]

    images = np.stack(images).astype(np.float32)
    intrinsics = np.loadtxt(os.path.join(path, 'intrinsics.txt'))

    return images, intrinsics


def make_predictions(args):
    """ runs inference on the test images """

    np.random.seed(1234)
    cfg = config.cfg_from_file(args.cfg)

    deepv2d = DeepV2D(cfg, args.model, use_fcrn=args.fcrn, 
        mode=args.mode, is_calibrated=(not args.uncalibrated))

    with tf.Session() as sess:
        deepv2d.set_session(sess)

        test_path = 'data/nyu/nyu'
        test_paths = sorted(os.listdir(test_path))
        num_test = len(test_paths)

        predictions = []
        for test_id in tqdm(range(num_test)):
            images, intrinsics = load_test_sequence(os.path.join(test_path, test_paths[test_id]), args.n_frames)
            depth_predictions, _ = deepv2d(images, intrinsics, iters=args.n_iters)
        
            keyframe_depth = depth_predictions[0]
            keyframe_image = images[0]
            predictions.append(keyframe_depth.astype(np.float32))

            if args.viz:
                image_and_depth = vis.create_image_depth_figure(keyframe_image, keyframe_depth)
                cv2.imshow('image', image_and_depth/255.0)
                cv2.waitKey(10)

        return predictions


def evaluate(groundtruth, predictions):
    """ nyu evaluations """
    
    crop = [20, 459, 24, 615] # eigen crop
    gt_list = []
    pr_list = []

    num_test = len(predictions)
    for i in range(num_test):
        depth_gt = groundtruth[i]
        depth_pr = predictions[i]

        # crop and resize
        depth_pr = cv2.resize(depth_pr, (640, 480))
        depth_pr = depth_pr[crop[0]:crop[1], crop[2]:crop[3]]
        depth_gt = depth_gt[crop[0]:crop[1], crop[2]:crop[3]]

        # scale predicted depth to match gt
        scalor = eval_utils.compute_scaling_factor(depth_gt, depth_pr, min_depth=0.8, max_depth=10.0)
        depth_pr = scalor * depth_pr

        gt_list.append(depth_gt)
        pr_list.append(depth_pr)

    depth_results = eval_utils.compute_depth_errors(gt_list, pr_list)
    print(("{:>10}, "*len(depth_results)).format(*depth_results.keys()))
    print(("{:10.4f}, "*len(depth_results)).format(*depth_results.values()))

    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='cfgs/nyu.yaml', help='config file used to train the model')
    parser.add_argument('--mode', default='keyframe', help='config file used to train the model')
    parser.add_argument('--model', default='models/nyu.ckpt', help='path to model checkpoint')

    parser.add_argument('--prediction_file', default="nyu_pred", help='where to put predicted depths')
    parser.add_argument('--viz', action="store_true", help='display depth maps during inference')
    parser.add_argument('--fcrn', action="store_true", help='use single image depth initializiation')
    parser.add_argument('--scale', action="store_true", help='use single image depth initializiation')
    parser.add_argument('--n_frames', type=int, default=-1, help='number of video frames to use for reconstruction')
    parser.add_argument('--n_iters', type=int, default=8, help='number of video frames to use for reconstruction')
    parser.add_argument('--uncalibrated', action="store_true", help='run in uncalibrated mode')
    args = parser.parse_args()

    # run inference on the test images
    predictions = make_predictions(args)
    groundtruth = np.load('data/nyu/nyu_groundtruth.npy')

    # evaluate on NYUv2 test set
    evaluate(groundtruth, predictions)
    


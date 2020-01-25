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
import pickle

import eval_utils
from core import config
from deepv2d import DeepV2D
from data_stream.kitti import KittiRaw


def process_for_evaluation(depth, scale, crop):
    """ During training ground truth depths are scaled and cropped, we need to 
        undo this for evaluation """
    depth = (1.0/scale) * np.pad(depth, [[crop, 0], [0, 0]], 'mean')
    return depth

def make_predictions(args):
    """ Run inference over the test images """

    np.random.seed(1234)
    cfg = config.cfg_from_file(args.cfg)

    db = KittiRaw(args.dataset_dir)
    scale = db.args['scale']
    crop = db.args['crop']
 
    deepv2d = DeepV2D(cfg, args.model, use_fcrn=False, mode='keyframe')

    with tf.Session() as sess:
        deepv2d.set_session(sess)

        predictions = []
        for (images, intrinsics) in db.test_set_iterator():
            depth_predictions, _ = deepv2d(images, intrinsics, iters=args.n_iters)
        
            keyframe_depth = depth_predictions[0]
            keyframe_image = images[0]

            pred = process_for_evaluation(keyframe_depth, scale, crop)
            predictions.append(pred.astype(np.float32))

            if args.viz:
                image_and_depth = vis.create_image_depth_figure(keyframe_image, keyframe_depth)
                cv2.imshow('image', image_and_depth/255.0)
                cv2.waitKey(10)

        return predictions


def evaluate(groundtruth, predictions, min_depth=1e-3, max_depth=80):

    depth_results = {}
    for (t_id, depth_gt) in groundtruth:
        depth_pr = predictions[t_id]
        ht, wd = depth_gt.shape[:2]

        depth_pr = cv2.resize(depth_pr, (wd, ht))
        depth_pr = np.clip(depth_pr, min_depth, max_depth)

        mask = np.logical_and(depth_gt > min_depth,
                              depth_gt < max_depth)

        # crop used by Garg ECCV16 to reproduce Eigen NIPS14 results
        # if used on gt_size 370x1224 produces a crop of [-218, -3, 44, 1180]
        crop = np.array([0.40810811 * ht, 0.99189189 * ht,
                         0.03594771 * wd, 0.96405229 * wd])
        
        crop = crop.astype(np.int32)
        crop_mask = np.zeros(mask.shape)
        crop_mask[crop[0]:crop[1],crop[2]:crop[3]] = 1
        mask = np.logical_and(mask, crop_mask)

        depth_gt = depth_gt[mask]
        depth_pr = depth_pr[mask]

        depth_pr = np.median(depth_gt/depth_pr) * depth_pr
        depth_metrics = eval_utils.compute_depth_errors(
            depth_gt, depth_pr, min_depth=min_depth, max_depth=max_depth)

        if len(depth_results) == 0:
            for dkey in depth_metrics:
                depth_results[dkey] = []

        for dkey in depth_metrics:
            depth_results[dkey].append(depth_metrics[dkey])


    # aggregate results
    for dkey in depth_results:
        depth_results[dkey] = np.mean(depth_results[dkey])

    print(("{:>10}, "*len(depth_results)).format(*depth_results.keys()))
    print(("{:10.4f}, "*len(depth_results)).format(*depth_results.values()))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='cfgs/kitti.yaml', help='config file used to train the model')
    parser.add_argument('--model', default='models/kitti.ckpt', help='path to model checkpoint')
    parser.add_argument('--dataset_dir', default='data/kitti/raw', help='config file used to train the model')
    parser.add_argument('--viz', action="store_true", help='display depth maps during inference')
    parser.add_argument('--n_iters', type=int, default=5, help='number of video frames to use for reconstruction')
    args = parser.parse_args()

    # run inference on the test images
    predictions = make_predictions(args)
    groundtruth = pickle.load(open('data/kitti/kitti_groundtruth.pickle', 'rb'))

    # evaluate on KITTI test set
    evaluate(groundtruth, predictions)
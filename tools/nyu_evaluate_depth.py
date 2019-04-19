import numpy as np
import scipy.io as sio

import pickle
import h5py
import cv2
import sys
import os
import argparse


# copied from https://github.com/lmb-freiburg/demon
def scale_invariant(depth1,depth2):
    """
    Computes the scale invariant loss based on differences of logs of depth maps.
    Takes preprocessed depths (no nans, infs and non-positive values)
    depth1:  one depth map
    depth2:  another depth map
    Returns:
        scale_invariant_distance
    """
    # sqrt(Eq. 3)
    assert(np.all(np.isfinite(depth1) & np.isfinite(depth2) & (depth1 > 0) & (depth2 > 0)))
    log_diff = np.log(depth1) - np.log(depth2)
    num_pixels = float(log_diff.size)

    if num_pixels == 0:
        return np.nan
    else:
        return np.sqrt(np.sum(np.square(log_diff)) / num_pixels - np.square(np.sum(log_diff)) / np.square(num_pixels))


def load_gt_depths(matfile, splitsfile):
    f = h5py.File(matfile, 'r')
    depths = np.array(f['depths'])
    splits = sio.loadmat(splitsfile)

    depth_gt = []
    for idx in splits['testNdxs']:
        depth_gt.append(depths[idx[0]-1].flatten(order='F').reshape(480, 640))

    return np.array(depth_gt)


def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25   ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred)**2) / gt)

    return abs_rel, rmse, rmse_log, a1, a2, a3



def main(args):

    depth_gt = load_gt_depths(args.gt_file, args.split_file)
    depth_pred = np.load(args.pred_file)

    depth_resized = []
    for depth in depth_pred:
        depth_resized.append(cv2.resize(depth, (640, 480)))

    depth_pred = np.array(depth_resized).astype("float64")
    depth_gt = np.array(depth_gt).astype("float64")

    sc = []
    for (d1, d2) in zip(depth_pred, depth_gt):
        sc.append(scale_invariant(d1, d2))

    print("scale_invariant: {:10.4f}".format(np.mean(sc)))

    abs_rel, rmse, rmse_log, a1, a2, a3 =\
        compute_errors(depth_gt.reshape(-1), depth_pred.reshape(-1))

    print("Absolute Error")
    print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format('abs_rel', 'rms', 'log_rms', 'a1', 'a2', 'a3'))
    print("{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}".format(abs_rel.mean(), rmse.mean(), rmse_log.mean(),  a1.mean(), a2.mean(), a3.mean()))


    for i in range(depth_pred.shape[0]):
        scalor = np.median(depth_gt[i]) / np.median(depth_pred[i])
        depth_pred[i] *= scalor

    abs_rel, rmse, rmse_log, a1, a2, a3 =\
        compute_errors(depth_gt.reshape(-1), depth_pred.reshape(-1))

    print("\nMedian Matching")
    print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format('abs_rel', 'rms', 'log_rms', 'a1', 'a2', 'a3'))
    print("{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}".format(abs_rel.mean(), rmse.mean(), rmse_log.mean(),  a1.mean(), a2.mean(), a3.mean()))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_file', help='path to model checkpoint')
    parser.add_argument('--split_file', help='')
    parser.add_argument('--pred_file', default='nyu_pred.npy', help='config file used to train the model')
    args = parser.parse_args()

    main(args)

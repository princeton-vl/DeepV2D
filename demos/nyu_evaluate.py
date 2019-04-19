import sys
sys.path.append('lib')

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import cv2
import os
import time
import argparse
import glob

from core import config
from deepv2d import DeepV2D


def load_test_sequence(path, n_frames):
    images = []
    for imfile in sorted(glob.glob(os.path.join(path, "*.png"))):
        images.append(cv2.imread(imfile))

    n = len(images)
    ix = [0] + np.random.choice(range(1,n), n_frames, replace=False).tolist()
    images = [images[i] for i in ix]

    images = np.array(images, dtype=np.uint8)
    intrinsics = np.loadtxt(os.path.join(path, 'intrinsics.txt'))

    test_blob = {
        'images': images,
        'intrinsics': intrinsics
    }

    return test_blob


def main(args):

    np.random.seed(1234)
    cfg = config.cfg_from_file(args.cfg)
    INPUT_DIMS = [args.n_frames+1, cfg.INPUT.HEIGHT, cfg.INPUT.WIDTH]
    net = DeepV2D(INPUT_DIMS, cfg, use_fcrn=args.fcrn_init)

    init_mode = 'constant'
    predictions = []
    with tf.Session() as sess:
        net.restore(sess, args.model)

        if args.fcrn_init:
            net.set_fcrn_weights(sess)
            init_mode = 'fcrn'

        if args.viz:
            fig, (ax1, ax2) = plt.subplots(1, 2)

        test_path = 'nyu_data/nyu'
        for sequence in sorted(os.listdir(test_path)):
            data_blob = load_test_sequence(os.path.join(test_path, sequence), args.n_frames)
            depth_predictions = net.forward(data_blob, iters=args.n_iters, init_mode=init_mode)
            depth = np.squeeze(depth_predictions[-1])

            if args.viz:
                ax1.cla()
                ax2.cla()
                keyframe = data_blob['images'][0]
                keyframe = keyframe[...,::-1]/255.0
                ax1.imshow(keyframe)
                ax2.imshow(depth)
                ax1.axis('off')
                ax2.axis('off')
                plt.pause(0.05)

            predictions.append(depth)

    predictions = np.stack(predictions, axis=0)
    np.save(args.prediction_file, predictions)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='cfgs/nyu.yaml', help='config file used to train the model')
    parser.add_argument('--model', default='models/nyu/nyu.ckpt', help='path to model checkpoint')
    parser.add_argument('--prediction_file', default="nyu_pred", help='where to put predicted depths')
    parser.add_argument('--viz', action="store_true", help='display depth maps during inference')
    parser.add_argument('--fcrn_init', action="store_true", help='use single image depth initializiation')
    parser.add_argument('--n_frames', type=int, default=7, help='number of video frames to use for reconstruction')
    parser.add_argument('--n_iters', type=int, default=3, help='number of video frames to use for reconstruction')
    args = parser.parse_args()

    main(args)

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


INPUT_DIMS = [5, 192, 1088]

def load_test_sequence(path):
    images = []
    for imfile in sorted(glob.glob(os.path.join(path, "*.png"))):
        images.append(cv2.imread(imfile))

    images = np.array(images, dtype=np.uint8)
    intrinsics = np.loadtxt(os.path.join(path, 'intrinsics.txt'))

    test_blob = {
        'images': images,
        'intrinsics': intrinsics
    }

    return test_blob


def main(args):
    cfg = config.cfg_from_file(args.cfg)
    net = DeepV2D(INPUT_DIMS, cfg)

    with tf.Session() as sess:
        net.restore(sess, args.model)

        data_blob = load_test_sequence(args.sequence)
        depths = net.forward(data_blob)

        # depth is scaled by 0.1 during training
        depth = 10*np.squeeze(depths[-1])

        fig, (ax1, ax2) = plt.subplots(2, 1)
        keyframe = data_blob['images'][0]
        keyframe = keyframe[...,::-1]/255.0
        ax1.imshow(keyframe)
        ax2.imshow(depth)
        ax1.axis('off')
        ax2.axis('off')
        plt.show()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='cfgs/kitti.yaml', help='config file used to train the model')
    parser.add_argument('--model', default='models/kitti/_stage_2.ckpt-120000', help='path to model checkpoint')
    parser.add_argument('--sequence', help='path to kitti sequence folder')
    args = parser.parse_args()

    main(args)

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
from data_stream.kitti import KittiRaw


INPUT_DIMS = [5, 192, 1088]

def process_for_evaluation(depth, scale, crop):
    depth = (1.0/scale) * np.pad(depth, [[crop, 0], [0, 0]], 'mean')
    return depth


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
    # we train DeepV2D on scaled and cropped images
    db = KittiRaw(args.dataset_dir)
    scale = db.args['scale']
    crop = db.args['crop']

    net = DeepV2D(INPUT_DIMS, cfg)


    predictions = []
    with tf.Session() as sess:
        net.restore(sess, args.model)

        if args.viz:
            fig, (ax1, ax2) = plt.subplots(2, 1)

        for data_blob in db.test_set_iterator():
            depths = net.forward(data_blob)
            depth = np.squeeze(depths[-1])

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

            depth = process_for_evaluation(depth, scale, crop)
            predictions.append(depth)


    predictions = np.array(predictions)
    np.save(args.prediction_file, predictions)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', help='config file used to train the model')
    parser.add_argument('--dataset_dir', help='path to kitti raw dataset')
    parser.add_argument('--model', help='path to model checkpoint')
    parser.add_argument('--prediction_file', default="kitti_pred", help='where to put predicted depths')
    parser.add_argument('--viz', action="store_true", help='vizualize depth maps during inference')
    args = parser.parse_args()

    main(args)

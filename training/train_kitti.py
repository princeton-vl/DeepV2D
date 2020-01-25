import sys
sys.path.append('deepv2d')

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import cv2
import os
import time
import random
import argparse

from core import config
from trainer import DeepV2DTrainer


def main(args):

    cfg = config.cfg_from_file(args.cfg)

    log_dir = os.path.join('logs/kitti', args.name)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    checkpoint_dir = os.path.join('checkpoints/kitti', args.name)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    tmp_dir = os.path.join('tmp/nyu', args.name)
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)

    cfg.LOG_DIR = log_dir
    cfg.CHECKPOINT_DIR = checkpoint_dir
    cfg.TMP_DIR = tmp_dir

    solver = DeepV2DTrainer(cfg)
    ckpt = None

    if args.restore is not None:
        solver.train(args.tfrecords, cfg, stage=2, restore_ckpt=args.restore, num_gpus=args.num_gpus)

    else:
        for stage in [1, 2]:
            ckpt = solver.train(args.tfrecords, cfg, stage=stage, ckpt=ckpt, num_gpus=args.num_gpus)
            tf.reset_default_graph()

if __name__ == '__main__':

    seed = 1234
    tf.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    parser = argparse.ArgumentParser()
    parser.add_argument('--name', help='name of your experiment')
    parser.add_argument('--cfg', default='cfgs/kitti.yaml', help='path to yaml config file')
    parser.add_argument('--tfrecords', default='datasets/kitti_train.tfrecords', help='path to tfrecords training file')
    parser.add_argument('--restore',  help='use restore checkpoint')
    parser.add_argument('--num_gpus', type=int, default=1, help='number of gpus to use')

    args = parser.parse_args()

    main(args)

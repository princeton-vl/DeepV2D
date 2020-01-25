"""
DeepV2D config system
    modeled after Faster-RCNN config system -
        https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/fast_rcnn/config.py
"""

import os
import os.path as osp
import numpy as np
import yaml
from yaml import Loader, Dumper
# `pip install easydict` if you don't have it
from easydict import EasyDict as edict

__C = edict()
# Consumers can get config by:
#   from fast_rcnn_config import cfg
cfg = __C

#
# Training options
#

__C.TMP_DIR = 'tmp'

__C.IS_TRAINING = True
__C.ONLINE_NORMALIZATION = False

__C.TRAIN = edict()

__C.INPUT = edict()

# input dimensions
__C.INPUT.FRAMES = 9
__C.INPUT.HEIGHT = 480
__C.INPUT.WIDTH = 640
__C.INPUT.SAMPLES = 3

# random scale augumentation
__C.INPUT.SCALES = [1.0]

# training parameters
__C.TRAIN.ITERS = [20000, 120000]
__C.TRAIN.BATCH = [4, 2]


__C.TRAIN.GT_POSE_ITERS = 10000
__C.TRAIN.LR = 0.001
__C.TRAIN.LR_DECAY = 0.9
__C.TRAIN.REGRESSOR_INIT = False
__C.TRAIN.RENORM = True
__C.TRAIN.CLIP_GRADS = True

__C.TRAIN.DEPTH_WEIGHT = 1.0

__C.STRUCTURE = edict()
__C.STRUCTURE.MIN_DEPTH = 0.1
__C.STRUCTURE.MAX_DEPTH = 8.0
__C.STRUCTURE.COST_VOLUME_DEPTH = 32

__C.STRUCTURE.RESCALE_IMAGES = False

# type of view aggregation to use (either 'avg' or 'concat')
__C.STRUCTURE.MODE = 'avg'

# number of stacked 3D hourglass modules to use
__C.STRUCTURE.HG_COUNT = 2

__C.STRUCTURE.TRAIN = edict()
# small smoothing loss over missing depth values
__C.STRUCTURE.TRAIN.SMOOTH_W = 0.02

__C.MOTION = edict()
# stack frames when estimating camera motion
__C.MOTION.STACK_FRAMES = False

__C.MOTION.IS_CALIBRATED = True

__C.MOTION.RESCALE_IMAGES = False

__C.MOTION.INTERNAL = 'matrix'

__C.MOTION.GN_STEPS = 2

__C.MOTION.EP_LMBDA = 100

__C.MOTION.LM_LMBDA = 0.0001

# FlowSE3 parameters
__C.MOTION.FLOWSE3 = edict()
__C.MOTION.FLOWSE3.ITER_COUNT = 3

# Levenberg-Marquardt dampening factor
__C.MOTION.FLOWSE3.LM_LAMBDA = 0.0

# Small delta added to diagonal to avoid singular matrix
__C.MOTION.FLOWSE3.EP_LAMBDA = 100.0

__C.MOTION.TRAIN = edict()

# penelize residual following FlowSE3 layer
__C.MOTION.TRAIN.RESIDUAL_WEIGHT = 0.01

# encourage larger weights
__C.MOTION.TRAIN.WEIGHT_REG = 0.01

# perturb input motion by small SE3 transform (form of augumentation)
__C.MOTION.TRAIN.DELTA = [0.025, 0.025, 0.025, 0.025, 0.025, 0.025]


def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.items():
        # a must specify keys that are in b
        if not k in b:
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(('Type mismatch ({} vs. {}) '
                                'for config key: {}').format(type(b[k]),
                                                            type(v), k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
        else:
            b[k] = v

def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.safe_load(f))

    _merge_a_into_b(yaml_cfg, __C)
    return __C


def cfg_from_list(cfg_list):
    """Set config keys via list (e.g., from command line)."""
    from ast import literal_eval
    assert len(cfg_list) % 2 == 0
    for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
        key_list = k.split('.')
        d = __C
        for subkey in key_list[:-1]:
            assert d.has_key(subkey)
            d = d[subkey]
        subkey = key_list[-1]
        assert subkey in d
        try:
            value = literal_eval(v)
        except:
            # handle the case when v is a string literal
            value = v
        assert type(value) == type(d[subkey]), \
            'type {} does not match original type {}'.format(
            type(value), type(d[subkey]))
        d[subkey] = value

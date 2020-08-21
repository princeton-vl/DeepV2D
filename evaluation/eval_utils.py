import numpy as np


""" Evaluation Utils
        gt: groundtruth
        pr: prediction
"""

# As described in Section 4 of the paper: "Since it is not possible to recover the absolute 
# scale of the scene through SfM, we report all results (both ours and all other approaches) 
# using scale matched depth."
def compute_scaling_factor(gt, pr, min_depth=0.5, max_depth=8.0):
    gt = np.array(gt, dtype=np.float64).reshape(-1)
    pr = np.array(pr, dtype=np.float64).reshape(-1)

    # only use valid depth values
    v = (gt > min_depth) & (gt < max_depth)
    return np.median(gt[v] / pr[v])

# copied from https://github.com/lmb-freiburg/demon
def scale_invariant(gt, pr):
    """
    Computes the scale invariant loss based on differences of logs of depth maps.
    Takes preprocessed depths (no nans, infs and non-positive values)
    depth1:  one depth map
    depth2:  another depth map
    Returns:
        scale_invariant_distance
    """
    gt = gt.reshape(-1)
    pr = pr.reshape(-1)

    v = gt > 0.1
    gt = gt[v]
    pr = pr[v]

    log_diff = np.log(gt) - np.log(pr)
    num_pixels = np.float32(log_diff.size)

    # sqrt(Eq. 3)
    return np.sqrt(np.sum(np.square(log_diff)) / num_pixels \
        - np.square(np.sum(log_diff)) / np.square(num_pixels))
        
def compute_pose_errors(gt, pr):
    # seperate rotations and translations
    R1, t1 = gt[:3, :3], gt[:3, 3]
    R2, t2 = pr[:3, :3], pr[:3, 3]

    costheta = (np.trace(np.dot(R1.T, R2))-1.0)/2.0
    costheta = np.minimum(costheta, 1.0)
    rdeg = np.arccos(costheta) * (180/np.pi)

    t1mag = np.sqrt(np.dot(t1,t1))
    t2mag = np.sqrt(np.dot(t2,t2))
    costheta = np.dot(t1,t2) / (t1mag*t2mag)
    tdeg = np.arccos(costheta) * (180/np.pi)

    # fit scales to translations
    a = np.dot(t1, t2) / np.dot(t2, t2)
    tcm = 100*np.sqrt(np.sum((t1-a*t2)**2, axis=-1))

    pose_errors = {
        "rot ang": rdeg,
        "trans ang": tdeg,
        "trans cm": tcm}

    return pose_errors


def compute_depth_errors(gt, pr, min_depth=0.1, max_depth=10.0):
    if isinstance(pr, list):
        scinv_list = []
        for i in range(len(gt)):
            scinv_list.append(scale_invariant(gt[i], pr[i]))
        scinv = np.mean(scinv_list)

        gt = np.stack(gt).astype(np.float32).reshape(-1)
        pr = np.stack(pr).astype(np.float32).reshape(-1)

    else:
        scinv = scale_invariant(gt, pr)

    # igore invalid depth values from evaluation
    v = (gt > min_depth) & (gt < max_depth)
    gt, pr = gt[v], pr[v]

    # just put all the metrics in the dict
    thresh = np.maximum((gt / pr), (pr / gt))
    a10 = (thresh < 1.10).mean() # a1,a2,a3 becoming saturated
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25** 2).mean()
    a3 = (thresh < 1.25** 3).mean()

    rmse = (gt - pr) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pr)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pr) / gt)
    log10 = np.mean(np.abs(np.log10(gt) - np.log10(pr)))

    sq_rel1 = np.mean(((gt - pr)**2) / gt)
    sq_rel2 = np.mean(((gt - pr)**2) / gt**2)

    depth_errors = {
        "sc-inv": scinv,
        "a10": a10,
        "a1": a1,
        "a2": a2,
        "a3": a3,
        "rmse": rmse,
        "log_rmse": rmse_log,
        "rel": abs_rel,
        "sq_rel1": sq_rel1,
        "sq_rel2": sq_rel2,
        "log10": log10}

    return depth_errors

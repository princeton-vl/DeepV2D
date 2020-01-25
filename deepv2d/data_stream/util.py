from scipy import interpolate
import numpy as np

def quat2rotm(q):
    """Convert quaternion into rotation matrix """
    x, y, z, s = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    r1 = np.stack([1-2*(y**2+z**2), 2*(x*y-s*z), 2*(x*z+s*y)], axis=1)
    r2 = np.stack([2*(x*y+s*z), 1-2*(x**2+z**2), 2*(y*z-s*x)], axis=1)
    r3 = np.stack([2*(x*z-s*y), 2*(y*z+s*x), 1-2*(x**2+y**2)], axis=1)
    return np.stack([r1, r2, r3], axis=1)

def pose_vec2mat(pvec, use_filler=True):
    """Convert quaternion vector represention to SE3 group"""
    t, q = pvec[:, 0:3], pvec[:, 3:7]
    R = quat2rotm(q)
    t = np.expand_dims(t, axis=-1)
    P = np.concatenate([R, t], axis=2)
    if use_filler:
        filler = np.array([0.0, 0.0, 0.0, 1.0]).reshape([1, 1, 4])
        filler = np.tile(filler, [pvec.shape[0], 1, 1])
        P = np.concatenate([P, filler], axis=1)
    return P

def inv_SE3(G):
    H = np.eye(4)
    Rt = G[0:3, 0:3].T
    H[0:3,0:3] = Rt
    H[0:3, 3] = -Rt.dot(G[0:3, 3])
    return H

def solve_SE3(A, B):
    return np.dot(B, inv_SE3(A))


from scipy import interpolate


def fill_depth(depth):
    x, y = np.meshgrid(np.arange(depth.shape[1]).astype("float32"),
                       np.arange(depth.shape[0]).astype("float32"))
    xx = x[depth > 0]
    yy = y[depth > 0]
    zz = depth[depth > 0]

    grid = interpolate.griddata((xx, yy), zz.ravel(),
                                (x, y), method='nearest')
    return grid

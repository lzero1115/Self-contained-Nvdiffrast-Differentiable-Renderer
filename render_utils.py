import numpy as np
import igl
import torch
import matplotlib.pyplot as plt
import itertools
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import distance_matrix
import scipy.optimize

# notice that the camera coordinate is same as in OpenGL!!! rather than for traditional computer vision
def rotation_matrix(axis, angle):
    """
    Builds a homogeneous coordinate rotation matrix along an axis

    Parameters
    ----------

    axis : str
        Axis of rotation, x, y, or z
    angle : float
        Rotation angle, in degrees
    """
    assert axis in 'xyz', "Invalid axis, expected x, y or z"
    mat = torch.eye(4, device='cuda')
    theta = np.deg2rad(angle)
    idx = 'xyz'.find(axis)
    mat[(idx+1)%3, (idx+1)%3] = np.cos(theta)
    mat[(idx+2)%3, (idx+2)%3] = np.cos(theta)
    mat[(idx+1)%3, (idx+2)%3] = -np.sin(theta)
    mat[(idx+2)%3, (idx+1)%3] = np.sin(theta)
    return mat

def translation_matrix(tr):
    """
    Builds a homogeneous coordinate translation matrix

    Parameters
    ----------

    tr : numpy.array
        translation value
    """
    mat = torch.eye(4, device='cuda')
    mat[:3,3] = torch.tensor(tr, device='cuda')
    return mat


def look_at(eye, center, up=torch.tensor([0.0, 1.0, 0.0])):
    # eye: camera position center: target position
    # Normalize the input vectors
    forward = center - eye
    forward = forward / torch.norm(forward, p=2)

    # Compute the right vector
    right = torch.cross(forward, up)
    right = right / torch.norm(right, p=2)

    # Recompute the orthogonal up vector
    up = torch.cross(right, forward)

    # Create the rotation matrix
    rotation = torch.eye(4)
    rotation[:3, :3] = torch.stack([right, up, -forward], dim=1)
    rotation = torch.transpose(rotation, 0, 1)

    # Create the translation matrix
    translation = torch.eye(4)
    translation[:3, 3] = -eye

    # Combine rotation and translation to form the homogeneous look_at matrix
    look_at_matrix = torch.mm(rotation, translation)

    return look_at_matrix

def persp_proj(fov_x=45, ar=1, near=0.1, far=100):
    """
    Build a perspective projection matrix.

    Parameters
    ----------
    fov_x : float
        Horizontal field of view (in degrees).
    ar : float
        Aspect ratio (w/h).
    near : float
        Depth of the near plane relative to the camera.
    far : float
        Depth of the far plane relative to the camera.
    """
    fov_x = np.deg2rad(fov_x)
    # # proj_mat = np.array([[-1.0 / np.tan(fov_rad / 2.0), 0, 0, 0],
    # #                   [0, np.float32(ar) / np.tan(fov_rad / 2.0), 0, 0],
    # #                   [0, 0, -(near + far) / (near-far), 2 * far * near / (near-far)],
    # #                   [0, 0, 1, 0]])
    # proj_mat = np.array([
    #     [-1.0 / np.tan(fov_rad / 2.0), 0, 0, 0],
    #     [0, np.float32(ar) / np.tan(fov_rad / 2.0), 0, 0],
    #     [0, 0, -(near + far) / (near - far), -1],
    #     [0, 0, -2 * far * near / (near - far), 0]
    # ])

    t = np.tan(fov_x / 2)
    # Define the perspective projection matrix elements
    m00 = 1 / t
    m11 = np.float32(ar) / t
    m22 = -(far + near) / (far - near)
    m23 = -2 * far * near / (far - near)
    m32 = -1
    proj_mat = torch.tensor([
        [m00, 0,   0,   0],
        [0,   m11, 0,   0],
        [0,   0,   m22, m23],
        [0,   0,   m32, 0]
    ], device='cuda',dtype=torch.float32)

    # proj = torch.tensor(proj_mat, device='cuda', dtype=torch.float32)
    return proj_mat
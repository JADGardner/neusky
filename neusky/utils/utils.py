import torch
from functools import singledispatch, update_wrapper
import numpy as np
import torch.nn.functional as F
from torch import sin, cos, atan2, acos

from typing import Literal
from pathlib import Path


def linear_to_sRGB(color, use_quantile=False):
    """Convert linear RGB to sRGB.
    
    Args:
        color: [..., 3]
        use_quantile: Whether to use the 98th quantile to normalise the color values.
        
        Returns:
            color: [..., 3]
    """
    if use_quantile:
        q = torch.quantile(color.flatten(), 0.98)
        color = color / q.expand_as(color)

    color = torch.where(
        color <= 0.0031308,
        12.92 * color,
        1.055 * torch.pow(torch.abs(color), 1 / 2.4) - 0.055,
    )
    color = torch.clamp(color, 0.0, 1.0)
    return color

def random_points_on_unit_sphere(num_points, cartesian=True):
    """
    Generate a random set of points on a unit sphere.

    :param num_points: number of points to generate
    :param cartesian: if True, return points in cartesian coordinates
    :return: (num_points, 2 or 3) tensor of points
    """
    # get random points in spherical coordinates
    theta = 2 * torch.pi * torch.rand(num_points)
    phi = torch.acos(2 * torch.rand(num_points) - 1)
    if cartesian:
        return torch.stack(sph2cart(theta, phi), dim=1)
    return torch.stack([theta, phi], dim=1)


def random_inward_facing_directions(num_directions, normals):
    # num_directions = scalar
    # normals = (N, 3)
    # returns (N, num_directions, 3)

    # For each normal get a random set of directions
    directions = random_points_on_unit_sphere(num_directions * normals.shape[0], cartesian=True)
    directions = directions.reshape(normals.shape[0], num_directions, 3)

    # remove any directions that are not in the hemisphere of the associated normal
    dot_products = torch.sum(normals.unsqueeze(1) * directions, dim=2)
    mask = dot_products < 0

    # negate the directions that are not in the hemisphere
    directions[mask] = -directions[mask]

    return directions


def ray_sphere_intersection(positions, directions, radius):
    """Ray sphere intersection"""
    # ray-sphere intersection
    # positions is the origins of the rays
    # directions is the directions of the rays [numbe]
    # radius is the radius of the sphere

    sphere_origin = torch.zeros_like(positions)
    radius = torch.ones_like(positions[..., 0]) * radius

    a = 1 # direction is normalized
    b = 2 * torch.einsum("ij,ij->i", directions, positions - sphere_origin)
    c = torch.einsum("ij,ij->i", positions - sphere_origin, positions - sphere_origin) - radius**2

    discriminant = b**2 - 4 * a * c

    t0 = (-b - torch.sqrt(discriminant)) / (2 * a)
    t1 = (-b + torch.sqrt(discriminant)) / (2 * a)

    # since we are inside the sphere we want the positive t
    t = torch.max(t0, t1)

    # now we need to point on the sphere that we intersected
    intersection_point = positions + t.unsqueeze(-1) * directions

    return intersection_point

def sph2cart(theta, phi):
    x = torch.sin(phi) * torch.cos(theta)
    y = torch.sin(phi) * torch.sin(theta)
    z = torch.cos(phi)
    return x, y, z


def cart2sph(x, y, z):
    r = torch.sqrt(x**2 + y**2 + z**2)
    theta = torch.atan2(y, x)
    phi = torch.acos(z / r)
    return theta, phi


def look_at_target(camera_positions, target_positions, up_vector=torch.tensor([0.0, 0.0, 1.0])):
    def normalize(vectors):
        return vectors / torch.norm(vectors, dim=-1, keepdim=True)
    
    # make sure up vector is same device
    up_vector = up_vector.type_as(camera_positions)

    forward_vectors = -normalize(target_positions - camera_positions)

    right_vectors = normalize(torch.cross(up_vector[None, :], forward_vectors))

    actual_up_vectors = normalize(torch.cross(forward_vectors, right_vectors))

    c2w_matrices = torch.zeros(*camera_positions.shape[:-1], 4, 4).to(camera_positions.device)
    c2w_matrices[..., :3, 0] = right_vectors
    c2w_matrices[..., :3, 1] = actual_up_vectors
    c2w_matrices[..., :3, 2] = forward_vectors
    c2w_matrices[..., :3, 3] = camera_positions
    c2w_matrices[..., 3, 3] = 1.0

    return c2w_matrices

def log_loss(y_true, y_pred):
    diff = torch.log(y_pred + 1e-6) - torch.log(y_true + 1e-6)
    return torch.mean(diff ** 2) - (torch.var(diff) / 2)

def rotation_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    """
    Return 3D rotation matrix for rotating around the given axis by the given angle.
    """
    axis = np.asarray(axis)
    axis = axis / np.sqrt(np.dot(axis, axis))
    a = np.cos(angle / 2.0)
    b, c, d = -axis * np.sin(angle / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    rotation = np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])
    # convert to pytorch
    rotation = torch.from_numpy(rotation).float()
    return rotation

def find_nerfstudio_project_root(start_dir: Path = Path(".")) -> Path:
    """
    Find the project root by searching for a '.root' file.
    """
    # Go up in the directory tree to find the root marker
    for path in [start_dir, *start_dir.parents]:
        if (path / 'nerfstudio').exists():
            return path
    # If we didn't find it, raise an error
    raise ValueError("Project root not found.")


def rot_z(gamma: torch.Tensor):
    return torch.tensor([
        [cos(gamma), -sin(gamma), 0],
        [sin(gamma), cos(gamma), 0],
        [0, 0, 1]
    ], dtype=gamma.dtype)
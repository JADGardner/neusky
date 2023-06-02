import torch
from functools import singledispatch, update_wrapper
import numpy as np
import torch.nn.functional as F


def sRGB(color):
    color = torch.where(
        color <= 0.0031308,
        12.92 * color,
        1.055 * torch.pow(torch.abs(color), 1 / 2.4) - 0.055,
    )
    color = torch.clamp(color, 0.0, 1.0)
    return color


### RENI Losses ###
def WeightedMSE(model_output, ground_truth, sineweight):
    MSE = (((model_output - ground_truth) ** 2) * sineweight).reshape(model_output.shape[0], -1).mean(1).sum(0)
    return MSE


def KLD(mu, log_var, Z_dims=1):
    kld = -0.5 * ((1 + log_var - mu.pow(2) - log_var.exp()).view(mu.shape[0], -1)).sum(1)
    kld /= Z_dims
    kld = kld.sum(0)
    return kld


def WeightedCosineSimilarity(model_output, ground_truth, sineweight):
    return (1 - (F.cosine_similarity(model_output, ground_truth, dim=1, eps=1e-20) * sineweight[:, 0]).mean(1)).sum(0)


def CosineSimilarity(model_output, ground_truth):
    return 1 - F.cosine_similarity(model_output, ground_truth, dim=1, eps=1e-20).mean()


class RENITrainLoss(object):
    def __init__(self):
        super().__init__()

    def __call__(self, inputs, targets, sineweight):
        loss = WeightedMSE(inputs, targets, sineweight)
        return loss


class RENIVADTrainLoss(object):
    def __init__(self, beta=1, Z_dims=None):
        super().__init__()
        self.beta = beta
        self.Z_dims = Z_dims

    def __call__(self, inputs, targets, sineweight, mu, log_var):
        mse = WeightedMSE(inputs, targets, sineweight)
        kld = self.beta * KLD(mu, log_var, self.Z_dims)
        loss = mse + kld

        return loss, mse, kld


class RENITestLoss(object):
    def __init__(self, alpha=1, beta=1):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def __call__(self, inputs, targets, sineweight, Z):
        mse = WeightedMSE(inputs, targets, sineweight)
        prior = self.alpha * torch.pow(Z, 2).sum()
        cosine = self.beta * WeightedCosineSimilarity(inputs, targets, sineweight)
        loss = mse + prior + cosine
        return loss, mse, prior, cosine


class RENITestLossMask(object):
    def __init__(self, alpha=1.0, beta=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.mse = torch.nn.MSELoss(reduction="mean")

    def __call__(self, inputs, targets, mask, Z):
        inputs = inputs * mask
        targets = targets * mask
        mse = self.mse(inputs, targets)
        prior = self.alpha * torch.pow(Z, 2).sum()
        cosine = self.beta * CosineSimilarity(inputs, targets)
        loss = mse + prior + cosine
        return loss


def methdispatch(func):
    """
    A decorator that allows the defining of a function (or method)
    with multiple dispatch based on the type of one of its arguments.

    The decorator uses the `singledispatch` function from the `functools`
    module to create the dispatcher. The new function will call the
    original function, but with a different implementation based on
    the type of the second argument (`args[1]`). The `register`
    attribute of the new function allows you to define additional
    implementations for the original function based on the type
    of the second argument.

    Args:
        func (callable): The function to be decorated. This should be a
        function that takes at least two arguments.

    Returns:
        callable: A new function that acts as a dispatcher for the
        original function. The new function will call the original
        function with a different implementation based on the type
        of the second argument.
    """
    dispatcher = singledispatch(func)

    def wrapper(*args, **kw):
        return dispatcher.dispatch(args[1].__class__)(*args, **kw)

    wrapper.register = dispatcher.register
    update_wrapper(wrapper, func)
    return wrapper


def get_directions(sidelen):
    """Generates a flattened grid of (x,y,z,...) coordinates in a range of -1 to 1.
    sidelen: int
    dim: int"""
    u = (torch.linspace(1, sidelen, steps=sidelen) - 0.5) / (sidelen // 2)
    v = (torch.linspace(1, sidelen // 2, steps=sidelen // 2) - 0.5) / (sidelen // 2)
    v_grid, u_grid = torch.meshgrid(v, u, indexing="ij")
    uv = torch.stack((u_grid, v_grid), -1)  # [sidelen/2,sidelen, 2]
    uv = uv.reshape(-1, 2)  # [sidelen/2*sidelen,2]
    theta = np.pi * (uv[:, 0] - 1)
    phi = np.pi * uv[:, 1]
    directions = torch.stack(
        (
            torch.sin(phi) * torch.sin(theta),
            torch.cos(phi),
            -torch.sin(phi) * torch.cos(theta),
        ),
        -1,
    ).unsqueeze(
        0
    )  # shape=[1, sidelen/2*sidelen, 3]
    return directions


# sine of the polar angle for compensation of irregular equirectangular sampling
def get_sineweight(sidelen):
    """Returns a matrix of sampling densites"""
    u = (torch.linspace(1, sidelen, steps=sidelen) - 0.5) / (sidelen // 2)
    v = (torch.linspace(1, sidelen // 2, steps=sidelen // 2) - 0.5) / (sidelen // 2)
    v_grid, u_grid = torch.meshgrid(v, u, indexing="ij")
    uv = torch.stack((u_grid, v_grid), -1)  # [sidelen/2, sidelen, 2]
    uv = uv.reshape(-1, 2)  # [sidelen/2*sidelen, 2]
    phi = np.pi * uv[:, 1]
    sineweight = torch.sin(phi)  # [sidelen/2*sidelen]
    sineweight = sineweight.unsqueeze(1).repeat(1, 3).unsqueeze(0)  # shape=[1, sidelen/2*sidelen, 3]
    return sineweight


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

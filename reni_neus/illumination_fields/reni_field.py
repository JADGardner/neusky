# Copyright 2022 The Nerfstudio Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Classic NeRF field"""

from functools import singledispatch, update_wrapper
from typing import Literal, Type
from dataclasses import dataclass, field
import wget
import zipfile
import os

import numpy as np
import torch
from torch import nn

from reni_neus.illumination_fields.base_illumination_field import IlluminationField, IlluminationFieldConfig
from reni_neus.utils.utils import sRGB

####################################################################
###################### ↓↓↓↓↓ UTILS ↓↓↓↓↓ ###########################
####################################################################


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


def invariant_representation(
    Z, D, equivariance: Literal["None", "SO2", "SO3"] = "SO2", conditioning: Literal["FiLM", "Concat"] = "Concat"
):
    """Generates an invariant representation from latent code Z and direction coordinates D.

    Args:
        Z (torch.Tensor): Latent code (B x ndims x 3)
        D (torch.Tensor): Direction coordinates (B x npix x 3)
        equivariance (str): Type of equivariance to use. Options are 'none', 'SO2', and 'SO3'
        conditioning (str): Type of conditioning to use. Options are 'Concat' and 'FiLM'

    Returns:
        torch.Tensor: Invariant representation (B x npix x 2 x ndims + ndims^2 + 2)
    """
    if equivariance == "None":
        innerprod = torch.bmm(D, torch.transpose(Z, 1, 2))
        z_input = Z.flatten(start_dim=1).unsqueeze(1).repeat(1, D.shape[1], 1)
        if conditioning == "FiLM":
            return innerprod, z_input
        if conditioning == "Concat":
            model_input = torch.cat((innerprod, z_input), 2)
            return model_input
        raise ValueError(f"Invalid conditioning type {conditioning}")

    if equivariance == "SO2":
        z_xz = torch.stack((Z[:, :, 0], Z[:, :, 2]), -1)
        d_xz = torch.stack((D[:, :, 0], D[:, :, 2]), -1)
        # Invariant representation of Z, gram matrix G=Z*Z' is size B x ndims x ndims
        G = torch.bmm(z_xz, torch.transpose(z_xz, 1, 2))
        # Flatten G and replicate for all pixels, giving size B x npix x ndims^2
        z_xz_invar = G.flatten(start_dim=1).unsqueeze(1).repeat(1, D.shape[1], 1)
        # innerprod is size B x npix x ndims
        innerprod = torch.bmm(d_xz, torch.transpose(z_xz, 1, 2))
        d_xz_norm = torch.sqrt(D[:, :, 0] ** 2 + D[:, :, 2] ** 2).unsqueeze(2)
        # Copy Z_y for every pixel to be size B x npix x ndims
        z_y = Z[:, :, 1].unsqueeze(1).repeat(1, innerprod.shape[1], 1)
        # Just the y component of D (B x npix x 1)
        d_y = D[:, :, 1].unsqueeze(2)
        if conditioning == "FiLM":
            model_input = torch.cat((d_xz_norm, d_y, innerprod), 2)  # [B, npix, 2 + ndims]
            conditioning_input = torch.cat((z_xz_invar, z_y), 2)  # [B, npix, ndims^2 + ndims]
            return model_input, conditioning_input
        if conditioning == "Concat":
            # model_input is size B x npix x 2 x ndims + ndims^2 + 2
            model_input = torch.cat((innerprod, z_xz_invar, d_xz_norm, z_y, d_y), 2)
            return model_input
        raise ValueError(f"Invalid conditioning type {conditioning}")

    if equivariance == "SO3":
        G = Z @ torch.transpose(Z, 1, 2)
        innerprod = torch.bmm(D, torch.transpose(Z, 1, 2))
        z_invar = G.flatten(start_dim=1).unsqueeze(1).repeat(1, D.shape[1], 1)
        if conditioning == "FiLM":
            return innerprod, z_invar
        if conditioning == "Concat":
            return torch.cat((innerprod, z_invar), 2)
        raise ValueError(f"Invalid conditioning type {conditioning}")


def kaiming_leaky_init(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        torch.nn.init.kaiming_normal_(m.weight, a=0.2, mode="fan_in", nonlinearity="leaky_relu")


def frequency_init(freq):
    def init(m):
        with torch.no_grad():
            if isinstance(m, nn.Linear):
                num_input = m.weight.size(-1)
                m.weight.uniform_(-np.sqrt(6 / num_input) / freq, np.sqrt(6 / num_input) / freq)

    return init


def first_layer_film_sine_init(m):
    with torch.no_grad():
        if isinstance(m, nn.Linear):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-1 / num_input, 1 / num_input)


class CustomMappingNetwork(nn.Module):
    def __init__(self, in_features, map_hidden_layers, map_hidden_dim, map_output_dim):
        super().__init__()

        self.network = []

        for _ in range(map_hidden_layers):
            self.network.append(nn.Linear(in_features, map_hidden_dim))
            self.network.append(nn.LeakyReLU(0.2, inplace=True))
            in_features = map_hidden_dim

        self.network.append(nn.Linear(map_hidden_dim, map_output_dim))

        self.network = nn.Sequential(*self.network)

        self.network.apply(kaiming_leaky_init)
        with torch.no_grad():
            self.network[-1].weight *= 0.25

    def forward(self, z):
        frequencies_offsets = self.network(z)
        frequencies = frequencies_offsets[..., : torch.div(frequencies_offsets.shape[-1], 2, rounding_mode="floor")]
        phase_shifts = frequencies_offsets[..., torch.div(frequencies_offsets.shape[-1], 2, rounding_mode="floor") :]

        return frequencies, phase_shifts


class FiLMLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.layer = nn.Linear(input_dim, hidden_dim)

    def forward(self, x, freq, phase_shift):
        x = self.layer(x)
        freq = freq.expand_as(x)
        phase_shift = phase_shift.expand_as(x)
        return torch.sin(freq * x + phase_shift)


class UnMinMaxNormlise(object):
    """Undo the minmax normalisation

    Args:
        minmax (tuple): min and max values

    Returns:
        torch.Tensor: unnormalised tensor
    """

    def __init__(self, minmax):
        self.minmax = minmax

    def __call__(self, x):
        x = 0.5 * (x + 1) * (self.minmax[1] - self.minmax[0]) + self.minmax[0]
        x = torch.exp(x)
        return x


class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                self.linear.weight.uniform_(
                    -np.sqrt(6 / self.in_features) / self.omega_0,
                    np.sqrt(6 / self.in_features) / self.omega_0,
                )

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))


class RENIVariationalAutoDecoder(nn.Module):
    def __init__(
        self,
        dataset_size,
        ndims,
        equivariance,
        hidden_features,
        hidden_layers,
        out_features,
        last_layer_linear,
        output_activation,
        first_omega_0,
        hidden_omega_0,
        minmax,
        fixed_decoder,
    ):
        super().__init__()
        # set all hyperaparameters from config
        self.dataset_size = dataset_size
        self.ndims = ndims
        self.equivariance = equivariance
        self.hidden_features = hidden_features
        self.hidden_layers = hidden_layers
        self.out_features = out_features
        self.last_layer_linear = last_layer_linear
        self.output_activation = output_activation
        self.first_omega_0 = first_omega_0
        self.hidden_omega_0 = hidden_omega_0
        self.minmax = minmax
        self.fixed_decoder = fixed_decoder

        if self.equivariance == "None":
            self.in_features = self.ndims * 3 + self.ndims
        elif self.equivariance == "SO2":
            self.in_features = 2 * self.ndims + self.ndims * self.ndims + 2
        elif self.equivariance == "SO3":
            self.in_features = self.ndims + self.ndims * self.ndims

        self.init_latent_codes(self.dataset_size, self.ndims, self.fixed_decoder)

        self.unnormalise = UnMinMaxNormlise(self.minmax)

        self.net = []

        self.net.append(
            SineLayer(
                self.in_features,
                self.hidden_features,
                is_first=True,
                omega_0=self.first_omega_0,
            )
        )

        for _ in range(self.hidden_layers):
            self.net.append(
                SineLayer(
                    self.hidden_features,
                    self.hidden_features,
                    is_first=False,
                    omega_0=self.hidden_omega_0,
                )
            )

        if self.last_layer_linear:
            final_linear = nn.Linear(self.hidden_features, self.out_features)

            with torch.no_grad():
                final_linear.weight.uniform_(
                    -np.sqrt(6 / self.hidden_features) / self.hidden_omega_0,
                    np.sqrt(6 / self.hidden_features) / self.hidden_omega_0,
                )

            self.net.append(final_linear)
        else:
            self.net.append(
                SineLayer(
                    self.hidden_features,
                    self.out_features,
                    is_first=False,
                    omega_0=self.hidden_omega_0,
                )
            )

        if self.output_activation == "exp":
            self.net.append(torch.exp)
        elif self.output_activation == "tanh":
            self.net.append(nn.Tanh())

        self.net = nn.Sequential(*self.net)

        if self.fixed_decoder:
            for param in self.net.parameters():
                param.requires_grad = False

    def get_Z(self):
        return self.mu

    def sample_latent(self, idx):
        """Sample the latent code at a given index

        Args:
        idx (int): Index of the latent variable to sample

        Returns:
        tuple (torch.Tensor, torch.Tensor, torch.Tensor): A tuple containing the sampled latent variable, the mean of the latent variable and the log variance of the latent variable
        """
        if torch.is_grad_enabled():
            mu = self.mu[idx, :, :]
            log_var = self.log_var[idx, :, :]
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            sample = mu + (eps * std)
            return sample, mu, log_var
        else:
            return self.mu[idx, :, :], self.mu[idx, :, :], self.log_var[idx, :, :]

    def init_latent_codes(self, dataset_size, ndims, fixed_decoder=True):
        self.log_var = torch.nn.Parameter(torch.normal(-5, 1, size=(dataset_size, ndims, 3)))
        if fixed_decoder:
            self.mu = nn.Parameter(torch.zeros(dataset_size, ndims, 3))
            self.log_var.requires_grad = False
        else:
            self.mu = nn.Parameter(torch.randn((dataset_size, ndims, 3)))

    def load_state_dict(self, state_dict, strict: bool = True):
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("model."):
                new_state_dict[k[6:]] = v

        if self.fixed_decoder:
            net_state_dict = {}
            for key in new_state_dict.keys():
                if key.startswith("net."):
                    net_state_dict[key[4:]] = new_state_dict[key]
            self.net.load_state_dict(net_state_dict, strict=strict)
        else:
            super().load_state_dict(new_state_dict, strict=strict)

    @methdispatch
    def forward(self, x, directions):
        raise NotImplementedError(
            "x must be either an int (idx), torch.Tensor (idxs or latent codes) or a list of ints (idxs)"
        )

    @forward.register
    def _(self, idx: int, directions):
        assert len([idx]) == directions.shape[0]
        if self.fixed_decoder:
            Z = self.mu[[idx], :, :]
        else:
            Z, _, _ = self.sample_latent([idx])
        x = invariant_representation(Z, directions, equivariance=self.equivariance, conditioning="Concat")
        return self.net(x)

    @forward.register
    def _(self, idx: list, directions):
        assert len(idx) == directions.shape[0]
        if self.fixed_decoder:
            Z = self.mu[idx, :, :]
        else:
            Z, _, _ = self.sample_latent(idx)
        x = invariant_representation(Z, directions, equivariance=self.equivariance, conditioning="Concat")
        return self.net(x)

    @forward.register
    def _(self, x: torch.Tensor, directions):
        if len(x.shape) == 1:
            idx = x
            if self.fixed_decoder:
                Z = self.mu[idx, :, :]
            else:
                Z, _, _ = self.sample_latent(idx)
            x = invariant_representation(Z, directions, equivariance=self.equivariance, conditioning="Concat")
        elif len(x.shape) == 4:
            Z = x
            raise NotImplementedError("Invariant representation for 4D tensors not implemented")
            x = self.InvariantRepresentation4D(Z, directions)
        else:
            Z = x
            x = invariant_representation(Z, directions, equivariance=self.equivariance, conditioning="Concat")
        return self.net(x)
    

def download_weights():
    """Downloads the weights for the RENI model"""
    download_folder = "/workspace/reni_neus/checkpoints/reni_weights"

    url = "https://www.dropbox.com/s/3zt9c3864e8936r/RENI_Pretrained_Weights.zip?dl=1"

    if not os.path.exists(download_folder):
        os.makedirs(download_folder)

    output = os.path.join(download_folder, "RENI_Pretrained_Weights.zip")

    if not os.path.exists(output):
        print("Downloading RENI weights...")
        wget.download(url, out=output)

    # if not extracted already
    if not os.path.exists(os.path.join(download_folder, "latent_dim_36_net_5_256_vad_cbc_tanh_hdr")):
        print("Extracting RENI weights...")
        with zipfile.ZipFile(output, "r") as zip_ref:
            zip_ref.extractall(download_folder)


def get_reni_field(chkpt_path, num_latent_codes, latent_dim, fixed_decoder=True):
    """Returns a RENI model from a PyTorch Lightning checkpoint

    Args:
        chkpt_path (str): path to PyTorch Lightning checkpoint
        num_latent_codes (int): number of latent codes to use

    Returns:
        nn.Module: The initalised RENI model
    """
    download_weights()
    chkpt = torch.load(chkpt_path)
    config = chkpt["hyper_parameters"]["config"]

    ### MODEL ###
    conditioning = config.RENI.CONDITIONING
    model_type = config.RENI.MODEL_TYPE
    equivariance = config.RENI.EQUIVARIANCE
    latent_dim = config.RENI.LATENT_DIMENSION
    hidden_layers = config.RENI.HIDDEN_LAYERS
    hidden_features = config.RENI.HIDDEN_FEATURES
    out_features = config.RENI.OUT_FEATURES
    last_layer_linear = config.RENI.LAST_LAYER_LINEAR
    output_activation = config.RENI.OUTPUT_ACTIVATION
    first_omega_0 = config.RENI.FIRST_OMEGA_0
    hidden_omega_0 = config.RENI.HIDDEN_OMEGA_0

    # model trained on minmax normalised log(HDR) images where minmax is a function
    # of the training dataset. We need to unnormalise the output of the model
    # for use in downstream tasks.
    minmax = [-18.0536, 11.4633]  # this is the minmax of the training dataset just taken from the config

    # create a model with the same hyperparameters as the checkpoint
    # but setting 'dataset_size' to the required number of latent
    # codes for your downstream task and 'fixed_decoder' to True
    if conditioning == "Cond-by-Concat":
        if model_type == "VariationalAutoDecoder":
            model = RENIVariationalAutoDecoder(
                num_latent_codes,
                latent_dim,
                equivariance,
                hidden_features,
                hidden_layers,
                out_features,
                last_layer_linear,
                output_activation,
                first_omega_0,
                hidden_omega_0,
                minmax,
                fixed_decoder=fixed_decoder,
            )
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    model.load_state_dict(chkpt["state_dict"])
    return model


# Field related configs
@dataclass
class RENIFieldConfig(IlluminationFieldConfig):
    """Configuration for model instantiation"""

    _target: Type = field(default_factory=lambda: RENIField)
    """target class to instantiate"""
    checkpoint_path: str = "checkpoints/"
    """path to PyTorch Lightning checkpoint"""
    latent_dim: int = 36
    """size of latent code to use, checkpoints provided for 36 and 49"""
    fixed_decoder: bool = True
    """whether to use the fixed decoder, not optimised during training"""
    optimise_exposure_scale: bool = False
    """whether to train a per-illumination scale parameter"""


class RENIField(IlluminationField):
    """RENI Field

    Args:
        chkpt_path (str): path to PyTorch Lightning checkpoint
        num_latent_codes (int): number of latent codes to use
        fixed_decoder (bool, optional): whether to use the fixed decoder. Defaults to True.
    """

    config: RENIFieldConfig

    def __init__(
        self,
        config: RENIFieldConfig,
        num_latent_codes: int,
    ):
        super().__init__()
        self.num_latent_codes = num_latent_codes
        self.chkpt_path = config.checkpoint_path
        self.latent_dim = config.latent_dim
        self.fixed_decoder = config.fixed_decoder
        self.exposure_scale = config.optimise_exposure_scale

        self.reni = get_reni_field(self.chkpt_path, self.num_latent_codes, self.latent_dim, self.fixed_decoder)

        if self.exposure_scale:
            self.scale = nn.Parameter(torch.ones(self.num_latent_codes))

    def reset_latents(self):
        """Resets the latent codes to random values"""
        self.reni.get_Z().data = torch.zeros_like(self.reni.get_Z().data)

    def get_latents(self):
        return self.reni.get_Z()

    def set_no_grad(self):
        # TODO (james): make generic for type of reni
        self.reni.mu.requires_grad = False

    def get_outputs(self, unique_indices, inverse_indices, directions, rotation, illumination_type):
        """Computes and returns the HDR illumination colours.

        Args:
            unique_indices: [rays_per_batch]
            inverse_indices: [rays_per_batch, samples_per_ray]
            directions: [num_directions, 3]

        Returns:
            light_colours: [num_rays * samples_per_ray, num_directions, 3]
            light_directions: [num_rays * samples_per_ray, num_directions, 3]
        """
        Z, _, _ = self.reni.sample_latent(unique_indices)  # [unique_indices, ndims, 3]

        if rotation is not None:
            rotation = rotation.type_as(Z)
            Z = torch.matmul(Z, rotation)  # [unique_indices, ndims, 3]
                
        if illumination_type == "illumination":
            # convert directions to RENI coordinate system
            light_directions = torch.stack([-directions[:, 0], directions[:, 2], directions[:, 1]], dim=1)
            light_directions = directions.unsqueeze(0).repeat(Z.shape[0], 1, 1).to(Z.device)  # [unique_indices, D, 3]
            light_colours = self.reni(Z, light_directions)  # [unique_indices, D, 3]
            light_colours = self.reni.unnormalise(light_colours)  # undo reni scaling between -1 and 1
            if self.exposure_scale:
                s = self.scale[unique_indices].view(-1, 1, 1)  # [unique_indices, 1, 1]
                light_colours = light_colours * s
            light_colours = light_colours[inverse_indices]  # [num_rays, samples_per_ray, D, 3]
            light_colours = light_colours.reshape(-1, directions.shape[0], 3)  # [num_rays * samples_per_ray, D, 3]
            # convert directions back to nerfstudio coordinate system
            light_directions = torch.stack(
                [-light_directions[:, :, 0], light_directions[:, :, 2], light_directions[:, :, 1]], dim=2
            )
            light_directions = light_directions[inverse_indices]  # [num_rays, samples_per_ray, D, 3]
            light_directions = light_directions.reshape(
                -1, directions.shape[0], 3
            )  # [num_rays * samples_per_ray, D, 3]
            return light_colours, light_directions
        elif illumination_type == "background":
            Z = Z[inverse_indices[:, 0], :, :]  # [num_rays, ndims, 3]
            light_directions = directions.unsqueeze(1)  # [num_rays, 1, 3]
            # convert directions to RENI coordinate system
            light_directions = torch.stack(
                [-light_directions[:, :, 0], light_directions[:, :, 2], light_directions[:, :, 1]], dim=2
            )
            light_colours = self.reni(Z, light_directions)  # [num_rays, 1, 3]
            light_colours = light_colours.squeeze(1)  # [num_rays, 3]
            light_colours = self.reni.unnormalise(light_colours)  # undo reni scaling between -1 and 1
            if self.exposure_scale:
                s = self.scale[inverse_indices[:, 0]].view(-1, 1)  # [unique_indices, 1]
                light_colours = light_colours * s
            light_colours = sRGB(light_colours)  # [num_rays, 1, 3]
            return light_colours, light_directions
        elif illumination_type == "envmap":
            light_colours = self.reni(Z, directions)  # [unique_indices, D, 3]
            light_colours = self.reni.unnormalise(light_colours)  # undo reni scaling between -1 and 1
            if self.exposure_scale:
                s = self.scale[unique_indices].view(-1, 1, 1)  # [unique_indices, 1, 1]
                light_colours = light_colours * s
            light_colours = sRGB(light_colours)  # [num_rays, 1, 3]
            return light_colours, None

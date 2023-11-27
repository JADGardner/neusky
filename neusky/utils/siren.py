import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class SineLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.

    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a
    # hyperparameter.

    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)

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
                    -np.sqrt(6 / self.in_features) / self.omega_0, np.sqrt(6 / self.in_features) / self.omega_0
                )

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))

    def forward_with_intermediate(self, input):
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate


class Siren(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features,
        hidden_layers,
        out_features,
        outermost_linear=False,
        first_omega_0=30,
        hidden_omega_0=30.0,
    ):
        super().__init__()

        self.net = []
        self.net.append(SineLayer(in_features, hidden_features, is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features, is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)

            with torch.no_grad():
                final_linear.weight.uniform_(
                    -np.sqrt(6 / hidden_features) / hidden_omega_0, np.sqrt(6 / hidden_features) / hidden_omega_0
                )

            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features, is_first=False, omega_0=hidden_omega_0))

        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        coords = coords.clone().detach().requires_grad_(True)  # allows to take derivative w.r.t. input
        output = self.net(coords)
        return output



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


class DDFFiLMSiren(nn.Module):
    def __init__(
        self,
        input_dim,
        mapping_network_input_dim,
        siren_hidden_features,
        siren_hidden_layers,
        mapping_network_features,
        mapping_network_layers,
        out_features,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.mapping_network_input_dim = mapping_network_input_dim
        self.siren_hidden_features = siren_hidden_features
        self.siren_hidden_layers = siren_hidden_layers
        self.mapping_network_features = mapping_network_features
        self.mapping_network_layers = mapping_network_layers
        self.out_features = out_features

        self.net = nn.ModuleList()

        self.net.append(FiLMLayer(self.input_dim, self.siren_hidden_features))

        for _ in range(self.siren_hidden_layers - 1):
            self.net.append(
                FiLMLayer(self.siren_hidden_features, self.siren_hidden_features)
            )

        self.final_layer = nn.Linear(self.siren_hidden_features, self.out_features)

        self.mapping_network = CustomMappingNetwork(
            self.mapping_network_input_dim,
            self.mapping_network_layers,
            self.mapping_network_features,
            (len(self.net)) * self.siren_hidden_features * 2,
        )

        self.net.apply(frequency_init(25))
        self.final_layer.apply(frequency_init(25))
        self.net[0].apply(first_layer_film_sine_init)

    def forward(self, x):
        # FiLM conditioning on positions
        # Siren input is directions
        positions = x[..., : self.mapping_network_input_dim]
        directions = x[..., self.mapping_network_input_dim :]
        frequencies, phase_shifts = self.mapping_network(positions)
        return self.forward_with_frequencies_phase_shifts(
            directions, frequencies, phase_shifts
        )
    
    def forward_with_frequencies_phase_shifts(self, x, frequencies, phase_shifts):
        frequencies = frequencies * 15 + 30

        for index, layer in enumerate(self.net):
            start = index * self.siren_hidden_features
            end = (index + 1) * self.siren_hidden_features
            x = layer(x, frequencies[..., start:end], phase_shifts[..., start:end])

        x = self.final_layer(x)
        return x
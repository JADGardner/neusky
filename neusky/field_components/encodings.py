import torch
import icosphere

from nerfstudio.field_components.encodings import Encoding

class IcosphereEncoding(Encoding):
    """Icosphere encoding

    Args:
        num_levels: Number of feature grids.
        max_icosphere_level: Maximum level of detail for the icosphere.
        features_per_vertex: Number of features per vertex.
        hash_init_scale: Value to initialize hash grid.
        implementation: Implementation of encoding. Fallback to torch if tcnn not available.
    """

    def __init__(
        self,
        num_levels: int = 16,
        max_icosphere_level: int = 3,
        features_per_vertex: int = 2,
        hash_init_scale: float = 0.001,
        implementation: Literal["tcnn", "torch"] = "tcnn",
    ) -> None:
        super().__init__(in_dim=3)
        self.num_levels = num_levels
        self.features_per_vertex = features_per_vertex
        self.level_vertices = []
        self.level_vertex_table = []

        for i in range(num_levels):
            icosphere_level = max_icosphere_level * (i + 1) // num_levels  # linearly distribute levels
            vertices, _ = icosphere.icosphere(icosphere_level)
            self.level_vertices.append(vertices)
            vertex_table = torch.rand(size=(len(vertices), features_per_vertex)) * 2 - 1
            vertex_table *= hash_init_scale
            self.level_vertex_table.append(nn.Parameter(vertex_table))

    def get_out_dim(self) -> int:
        return self.num_levels * self.features_per_vertex

    def find_closest_vertex(self, point: Float[Tensor, "3"], level: int) -> int:
        """Finds the closest vertex on the icosphere to the given point.

        Args:
            point: The point to find the closest vertex to.
            level: The level of the icosphere to find the vertex on.
        """
        distances = torch.sum((self.level_vertices[level] - point)**2, axis=-1)
        return torch.argmin(distances)

    def pytorch_fwd(self, in_tensor: Float[Tensor, "*bs input_dim"]) -> Float[Tensor, "*bs output_dim"]:
        """Forward pass using pytorch. Significantly slower than TCNN implementation."""

        assert in_tensor.shape[-1] == 3
        output = []
        for level in range(self.num_levels):
            closest_vertices = self.find_closest_vertex(in_tensor, level)
            encoded_value = self.level_vertex_table[level][closest_vertices]
            output.append(encoded_value)

        return torch.cat(output, dim=-1)  # [..., num_levels * features_per_vertex]

    def forward(self, in_tensor: Float[Tensor, "*bs input_dim"]) -> Float[Tensor, "*bs output_dim"]:
        if TCNN_EXISTS and self.tcnn_encoding is not None:
            return self.tcnn_encoding(in_tensor)
        return self.pytorch_fwd(in_tensor)

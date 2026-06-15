import pytest
import torch

from nerfstudio.field_components.spatial_distortions import SceneContraction
from neusky.fields.sdf_albedo_field import SDFAlbedoFieldConfig


pytest.importorskip("tinycudann")


def _make_field(*, inside_outside: bool = False, radius: float = 0.1):
    if not torch.cuda.is_available():
        pytest.skip("SDFAlbedoField hash-grid encoding requires CUDA for this test")

    cfg = SDFAlbedoFieldConfig(
        num_layers=2,
        hidden_dim=64,
        geo_feat_dim=32,
        num_layers_color=1,
        hidden_dim_color=32,
        use_grid_feature=True,
        weight_norm=True,
        inside_outside=inside_outside,
        analytic_sphere_init=True,
        analytic_sphere_radius=radius,
        log2_hashmap_size=12,
        max_res=64,
    )
    aabb = torch.tensor([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]], device="cuda")
    return cfg.setup(
        aabb=aabb,
        num_images=1,
        spatial_distortion=SceneContraction(order=2).cuda(),
    ).cuda().eval()


def test_analytic_sphere_init_matches_expected_sdf():
    radius = 0.1
    field = _make_field(radius=radius)
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.05, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [0.2, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.95, 0.0, 0.0],
        ],
        device="cuda",
    )

    with torch.no_grad():
        sdf = field.get_sdf_at_pos(points).squeeze(-1)

    expected = torch.linalg.norm(points, dim=-1) - radius
    assert torch.allclose(sdf, expected, atol=1e-5)
    assert sdf[0] < 0
    assert sdf[2].abs() < 1e-5
    assert torch.all(sdf[3:] > 0)


def test_analytic_sphere_init_keeps_normalized_camera_radii_outside():
    field = _make_field(radius=0.1)
    camera_like_points = torch.tensor(
        [
            [0.29, 0.0, 0.0],
            [0.0, 0.67, 0.0],
            [0.0, 0.0, 0.95],
        ],
        device="cuda",
    )

    with torch.no_grad():
        sdf = field.get_sdf_at_pos(camera_like_points).squeeze(-1)

    assert torch.all(sdf > 0)


def test_analytic_sphere_init_respects_inside_outside_sign_flip():
    field = _make_field(inside_outside=True, radius=0.1)
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [0.5, 0.0, 0.0],
        ],
        device="cuda",
    )

    with torch.no_grad():
        sdf = field.get_sdf_at_pos(points).squeeze(-1)

    expected = 0.1 - torch.linalg.norm(points, dim=-1)
    assert torch.allclose(sdf, expected, atol=1e-5)

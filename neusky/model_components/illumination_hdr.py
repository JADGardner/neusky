# Copyright 2024 James Gardner, University of York. All rights reserved.
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

"""Linear-HDR decoding for the RENI++ illumination prior.

NeuSky loads a pretrained RENI++ field as its illumination prior and treats the
field's RGB output as linear HDR radiance. Two prior parameterisations exist:

* Standard RENI/RENI++ (3-channel): the field output is turned into linear HDR
  by :meth:`RENIField.unnormalise` (log/min-max inverse of the training-time
  normalisation).
* Two-bracket RENI++ (6-channel, "complementary tonemapping"): the field emits
  two bounded [0, 1] brackets (extended-Reinhard LDR + log highlight) and linear
  HDR is reconstructed by :func:`reni.utils.tonemap.two_bracket_to_linear`.

This module centralises that choice in ONE place (:class:`IlluminationHDRDecode`)
so background colours, shading samples, and eval-latent fitting all convert
field outputs to linear HDR the same way. It is intentionally free of the
tinycudann / DDF stack so it can be imported and unit-tested on CPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import yaml

from reni.utils.tonemap import DEFAULT_M_LDR, DEFAULT_M_LOG, two_bracket_to_linear


class _ReniConfigLoader(yaml.SafeLoader):
    """YAML loader that reads a nerfstudio ``config.yml`` WITHOUT importing the
    python classes it references.

    nerfstudio serialises configs with ``!!python/object:...`` /
    ``!!python/name:...`` tags. The full ``yaml.Loader`` would import every
    referenced module (pulling in the tcnn/DDF stack), so instead we turn
    object nodes into plain dicts, name references into their dotted-path
    string, and apply/tuple nodes into lists. That is enough to read the
    dataparser and field scalars we care about.
    """


def _construct_python(loader: yaml.Loader, tag_suffix: str, node: yaml.Node) -> Any:
    if tag_suffix.startswith("name:") or tag_suffix.startswith("module:"):
        # A reference to a python callable/class; keep the dotted path string.
        return tag_suffix.split(":", 1)[1]
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node, deep=True)
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node, deep=True)
    try:
        return loader.construct_scalar(node)
    except Exception:
        return None


_ReniConfigLoader.add_multi_constructor("tag:yaml.org,2002:python/", _construct_python)


def _load_reni_run_config(config_path: Path) -> Optional[Dict[str, Any]]:
    """Parse a RENI run ``config.yml`` into nested dicts, or None if absent."""
    config_path = Path(config_path)
    if not config_path.exists():
        return None
    with config_path.open() as handle:
        return yaml.load(handle, Loader=_ReniConfigLoader)


@dataclass
class IlluminationHDRDecode:
    """How to turn a RENI illumination field's RGB output into linear HDR.

    Attributes:
        two_bracket: True for the 6-channel two-bracket parameterisation.
        m_ldr, m_log: extended-Reinhard / log bracket saturation points, read
            from the checkpoint's dataparser (defaults match RENI++ G7h).
        fixed_gauge: True when the fit used fixed-gauge luminance normalisation
            (99th-percentile luminance == target). Advisory; see notes on
            ``eval_scale`` semantics in the model.
        out_features: field output channel count (3 or 6).
        output_activation: field output activation ("None"/"sigmoid"/...), used
            to build the field to match the checkpoint architecture.
    """

    two_bracket: bool = False
    m_ldr: float = DEFAULT_M_LDR
    m_log: float = DEFAULT_M_LOG
    fixed_gauge: bool = False
    out_features: int = 3
    output_activation: str = "None"

    def to_linear_hdr(self, illumination_field, x: torch.Tensor) -> torch.Tensor:
        """Convert a RENI field RGB output ``x`` to linear HDR radiance.

        For the two-bracket case this exactly mirrors
        ``reni.models.reni_model.RENIModel._to_linear_hdr`` (bracket inversion +
        sigmoid blend at the tonemap module's default tau/delta). For the
        3-channel case it defers to ``illumination_field.unnormalise`` so the
        standard RENI/RENI++ path is byte-for-byte unchanged.
        """
        if self.two_bracket:
            return two_bracket_to_linear(x, m_ldr=self.m_ldr, m_log=self.m_log)
        return illumination_field.unnormalise(x)

    @classmethod
    def from_reni_run_config(cls, config_path: Path) -> "IlluminationHDRDecode":
        """Build a decode spec from a RENI run's ``config.yml``.

        Detection uses the dataparser's ``tonemap_targets`` flag (which also
        carries ``tonemap_m_ldr`` / ``tonemap_m_log``); the field's
        ``out_features`` is cross-checked and must be 6 when two-bracket. If the
        config is missing or lacks the tonemap keys (older 3-channel runs) the
        default 3-channel spec is returned, leaving the standard path unchanged.
        """
        config = _load_reni_run_config(config_path)
        if config is None:
            return cls()

        pipeline = config.get("pipeline", {}) or {}
        datamanager = pipeline.get("datamanager", {}) or {}
        dataparser = datamanager.get("dataparser", {}) or {}
        model = pipeline.get("model", {}) or {}
        field = model.get("field", {}) or {}

        two_bracket = bool(dataparser.get("tonemap_targets", False))
        out_features = int(field.get("out_features", 3))
        output_activation = str(field.get("output_activation", "None"))

        if two_bracket and out_features != 6:
            raise ValueError(
                "RENI config sets tonemap_targets=True but field.out_features="
                f"{out_features} (expected 6 for a two-bracket prior)."
            )

        return cls(
            two_bracket=two_bracket,
            m_ldr=float(dataparser.get("tonemap_m_ldr", DEFAULT_M_LDR)),
            m_log=float(dataparser.get("tonemap_m_log", DEFAULT_M_LOG)),
            fixed_gauge=bool(dataparser.get("fixed_gauge_normalisation", False)),
            out_features=out_features,
            output_activation=output_activation,
        )

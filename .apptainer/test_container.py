#!/usr/bin/env python3
"""Verify the NeuSky container is correctly configured.

Run inside the container:
    python .apptainer/test_container.py
"""

import sys

checks_passed = 0
checks_failed = 0


def check(name, fn):
    global checks_passed, checks_failed
    try:
        fn()
        print(f"  [PASS] {name}")
        checks_passed += 1
    except Exception as e:
        print(f"  [FAIL] {name}: {e}")
        checks_failed += 1


def check_cuda():
    import torch
    assert torch.cuda.is_available(), "CUDA not available"
    assert torch.cuda.get_device_name(0), "No GPU detected"


def check_tinycudann():
    import tinycudann  # noqa: F401


def check_nvdiffrast():
    import nvdiffrast  # noqa: F401


def check_nerfstudio():
    import nerfstudio  # noqa: F401


def check_neusky():
    import neusky  # noqa: F401


def check_reni():
    import reni  # noqa: F401


print("NeuSky container verification")
print("=" * 40)

check("PyTorch CUDA", check_cuda)
check("tiny-cuda-nn", check_tinycudann)
check("nvdiffrast", check_nvdiffrast)
check("nerfstudio", check_nerfstudio)
check("neusky", check_neusky)
check("ns_reni", check_reni)

print("=" * 40)
print(f"Results: {checks_passed} passed, {checks_failed} failed")

if checks_failed > 0:
    print("\nSome checks failed. If neusky/reni failed, run:")
    print("  .apptainer/apptainer.sh install")
    sys.exit(1)
else:
    print("\nAll checks passed!")

# SPDX-License-Identifier: Apache-2.0
"""Do not let selective or empty capture manifests turn a failed gate green."""

import subprocess
import sys

import pytest

from tensorrt_model_connect.families.cosyvoice3.validation.validate_flow_environment import validate_case_manifest


def cases():
    return [{"id": f"stress_{frames}_{masked}", "group": "original_stress"}
            for frames in (4, 17, 64, 128) for masked in (0, 1)]


def test_complete_stress_manifest():
    validate_case_manifest(cases(), has_acoustic=False)


@pytest.mark.parametrize("kind", ["empty", "missing", "duplicate", "unrecorded_acoustic"])
def test_incomplete_capture_cannot_pass(kind):
    values = cases()
    if kind == "empty":
        values = []
    elif kind == "missing":
        values.pop()
    elif kind == "duplicate":
        values[-1] = values[0]
    with pytest.raises(ValueError, match="every declared case"):
        validate_case_manifest(values, has_acoustic=kind == "unrecorded_acoustic")


@pytest.mark.parametrize("module", [
    "validation.validate_flow",
    "validation.validate_flow_pytorch",
    "validation.validate_flow_trajectory",
    "validation.validate_offline_flow",
    "validation.validate_flow_environment",
    "validation.audit_flow_fp64",
    "diagnostics.audit_flow_reference",
    "diagnostics.audit_acoustic_reference",
    "diagnostics.diagnose_flow_layers",
    "diagnostics.diagnose_flow_ops",
    "diagnostics.diagnose_attention_precision",
])
def test_help_without_gpu_dependencies(module):
    result = subprocess.run([sys.executable, "-m", "tensorrt_model_connect.families.cosyvoice3." + module, "--help"],
                            check=True, capture_output=True, text=True)
    assert "usage:" in result.stdout

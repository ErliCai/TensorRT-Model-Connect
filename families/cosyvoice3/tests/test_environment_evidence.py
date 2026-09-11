# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Do not let selective or empty capture manifests turn a failed gate green."""

import subprocess
import sys

import pytest

from families.cosyvoice3.diagnostics import (
    audit_acoustic_reference,
    audit_flow_reference,
    diagnose_attention_precision,
    diagnose_flow_layers,
    diagnose_flow_ops,
)
from families.cosyvoice3.validation import (
    audit_flow_fp64,
    validate_flow,
    validate_flow_environment,
    validate_flow_pytorch,
    validate_flow_trajectory,
    validate_offline_flow,
)
from families.cosyvoice3.validation.validate_flow_environment import validate_case_manifest


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
    validate_flow,
    validate_flow_pytorch,
    validate_flow_trajectory,
    validate_offline_flow,
    validate_flow_environment,
    audit_flow_fp64,
    audit_flow_reference,
    audit_acoustic_reference,
    diagnose_flow_layers,
    diagnose_flow_ops,
    diagnose_attention_precision,
])
def test_help_without_gpu_dependencies(module):
    result = subprocess.run([sys.executable, "-m", module.__name__, "--help"],
                            check=True, capture_output=True, text=True)
    assert "usage:" in result.stdout

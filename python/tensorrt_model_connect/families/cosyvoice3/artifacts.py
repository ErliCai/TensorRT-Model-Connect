# SPDX-License-Identifier: Apache-2.0
"""Family-local component hashing and atomic publication."""

import hashlib
import json
import os
from pathlib import Path
import tempfile


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_component(output, plan_name, plan, manifest):
    """Publish a complete component without updating an existing directory."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    if Path(plan_name).name != plan_name or not plan_name.endswith(".plan"):
        raise ValueError("plan_name must be a .plan filename")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".cosyvoice3-", dir=output.parent) as tmp:
        stage = Path(tmp) / "component"
        stage.mkdir()
        (stage / plan_name).write_bytes(plan)
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        if output.exists():
            raise FileExistsError(output)
        os.rename(stage, output)

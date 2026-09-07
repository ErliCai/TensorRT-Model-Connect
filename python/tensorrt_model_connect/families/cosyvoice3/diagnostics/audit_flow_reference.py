# SPDX-License-Identifier: Apache-2.0
"""Audit repeatability and FP32 SDPA sensitivity of the official reference.

No TensorRT graph runs here. These diagnostic comparisons do not replace or
relax the existing native-engine parity gates.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..artifacts import sha256_file
from ..config import FLOW_SHA256
from ..validation.validate_flow_pytorch import _cases, _official_dit, _ieee_fp32_reference, ATOL, RTOL


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--cosyvoice-source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel

    if sha256_file(args.model_dir / 'flow.pt') != FLOW_SHA256:
        raise ValueError('Checkpoint does not match pinned model')
    args.output.mkdir(exist_ok=False, parents=True)
    DiT, revision = _official_dit(args.cosyvoice_source)
    model = DiT(dim=1024, depth=22, heads=16, dim_head=64, ff_mult=2,
                mel_dim=80, mu_dim=80, spk_dim=80, out_channels=80,
                static_chunk_size=50, num_decoding_left_chunks=-1).eval()
    state = torch.load(args.model_dir / 'flow.pt', map_location='cpu', weights_only=True, mmap=True)
    prefix = 'decoder.estimator.'
    model.load_state_dict({k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}, strict=True)
    del state
    model.cuda()
    arrays, rows = {}, []
    with torch.inference_mode(), _ieee_fp32_reference(torch):
        for frames, masked, values in _cases():
            tensors = {k: torch.from_numpy(v).cuda() for k, v in values.items()}
            ref = model(**tensors, streaming=False).cpu().numpy()
            repeated = model(**tensors, streaming=False).cpu().numpy()
            with sdpa_kernel(SDPBackend.MATH):
                math = model(**tensors, streaming=False).cpu().numpy()
            label = f'{frames}_{int(masked)}'
            arrays[label + '_auto'] = ref
            arrays[label + '_math'] = math
            difference = np.abs(math - ref)
            row = {'frames': frames, 'masked': masked,
                   'repeat_exact': bool(np.array_equal(ref, repeated)),
                   'auto_vs_math_passed': bool(np.allclose(math, ref, atol=ATOL, rtol=RTOL)),
                   'max_abs_error': float(difference.max()),
                   'max_tolerance_ratio': float((difference / (ATOL + RTOL * np.abs(ref))).max())}
            rows.append(row)
            print(json.dumps(row), flush=True)
    np.savez(args.output / 'outputs.npz', **arrays)
    (args.output / 'report.json').write_text(json.dumps({
        'scope': 'official_pytorch_reference_self_consistency_not_engine_acceptance',
        'source_revision': revision, 'checkpoint_sha256': FLOW_SHA256,
        'torch_version': torch.__version__, 'cuda': torch.version.cuda,
        'gpu': torch.cuda.get_device_name(), 'cudnn_tf32': False, 'matmul_tf32': False,
        'atol': ATOL, 'rtol': RTOL, 'cases': rows,
    }, indent=2))


if __name__ == '__main__':
    main()

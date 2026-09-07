# SPDX-License-Identifier: Apache-2.0
"""Probe family graph operations with exact inputs captured from official DiT.

Unlike a whole-model layer trace, each probe receives the official input, so
its error excludes amplification of errors from preceding layers.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np

from .checkpoint_mapper import load_flow_weights
from .config import FlowConfig, ShapeProfile, FLOW_SHA256
from .__main__ import sha256_file
from .flow_builder import _Graph
from .flow_runtime import FlowEngine
from .validate_flow_pytorch import _cases, _official_dit, _ieee_fp32_reference, ATOL, RTOL


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--cosyvoice-source', type=Path, required=True)
    parser.add_argument('--component', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--frames', type=int, choices=(4, 17, 64, 128), default=17)
    parser.add_argument('--masked', action='store_true')
    args = parser.parse_args()
    import torch
    import tensorrt as trt

    args.output.mkdir(exist_ok=False, parents=True)
    if sha256_file(args.model_dir / 'flow.pt') != FLOW_SHA256:
        raise ValueError('Checkpoint does not match pinned model')
    values = next(v for n, m, v in _cases() if n == args.frames and m == args.masked)
    engine = FlowEngine(args.component)
    gpu_inputs = {k: torch.from_numpy(v).cuda() for k, v in values.items()}
    actual = [engine(**gpu_inputs).cpu().numpy() for _ in range(3)]
    print('same_plan_repeat_max_error', [float(np.abs(v - actual[0]).max()) for v in actual], flush=True)
    del engine, gpu_inputs
    gc.collect()
    torch.cuda.empty_cache()
    DiT, revision = _official_dit(args.cosyvoice_source)
    model = DiT(dim=1024, depth=22, heads=16, dim_head=64, ff_mult=2,
                mel_dim=80, mu_dim=80, spk_dim=80, out_channels=80,
                static_chunk_size=50, num_decoding_left_chunks=-1).eval()
    state = torch.load(args.model_dir / 'flow.pt', map_location='cpu', weights_only=True, mmap=True)
    prefix = 'decoder.estimator.'
    model.load_state_dict({k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}, strict=True)
    del state
    captured = {}
    operations = []

    def hook(key, kind):
        def capture(module, inputs, kwargs, output):
            source = inputs[0] if inputs else kwargs['x']
            captured[key + '_input'] = source.detach().cpu().numpy().copy()
            captured[key + '_expected'] = output.detach().cpu().numpy().copy()
            operations.append((key, kind))
        return capture

    handles = []
    for i in (0, 2, 18, 19, 20, 21):
        key = f'transformer_blocks.{i}'
        block = model.transformer_blocks[i]
        for suffix, module, kind in [
            ('.attn_norm.norm', block.attn_norm.norm, 'norm'),
            ('.attn_norm.linear', block.attn_norm.linear, 'linear'),
            ('.attn.to_q', block.attn.to_q, 'linear'),
            ('.attn.to_k', block.attn.to_k, 'linear'),
            ('.attn.to_v', block.attn.to_v, 'linear'),
            ('.attn', block.attn, 'attention'),
            ('.ff_norm', block.ff_norm, 'norm'),
            ('.ff.ff.0.0', block.ff.ff[0][0], 'linear'),
            ('.ff.ff.0.1', block.ff.ff[0][1], 'gelu'),
            ('.ff.ff.2', block.ff.ff[2], 'linear'),
        ]:
            handles.append(module.register_forward_hook(hook(key + suffix, kind), with_kwargs=True))
    model.cuda()
    with torch.inference_mode(), _ieee_fp32_reference(torch):
        expected = model(**{k: torch.from_numpy(v).cuda() for k, v in values.items()}, streaming=False).cpu().numpy()
    print('whole_model_max_error', float(np.abs(actual[0]-expected).max()), flush=True)
    for handle in handles:
        handle.remove()
    np.savez(args.output / 'official_inputs.npz', **captured, input_mask=values['mask'])
    del model
    gc.collect()
    torch.cuda.empty_cache()
    weights = load_flow_weights(args.model_dir, FlowConfig())
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    net = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    graph = _Graph(net, weights, FlowConfig(), ShapeProfile(4, 32, 128))
    input_values = {}
    freq = weights['rotary_embed.inv_freq']
    phases = np.repeat(np.arange(args.frames, dtype=np.float32)[:, None] * freq[None], 2, axis=-1)
    cos, sin = graph.const(np.cos(phases)[None]), graph.const(np.sin(phases)[None])
    mask = graph.const(values['mask'])
    for key, kind in operations:
        val = captured[key + '_input']
        inp = net.add_input(key + '_input', trt.float32, val.shape)
        input_values[inp.name] = val
        if kind == 'linear':
            out = graph.linear(inp, key)
        elif kind == 'norm':
            out = graph.norm(inp)
        elif kind == 'gelu':
            out = graph.gelu(inp)
        else:
            out = graph.attention(inp, mask, key, cos, sin)
        out.name = key + '_actual'
        net.mark_output(out)
    config = builder.create_builder_config()
    config.clear_flag(trt.BuilderFlag.TF32)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 256 * 1024 * 1024)
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    plan = builder.build_serialized_network(net, config)
    if plan is None:
        raise RuntimeError('Operation probe build failed')
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(plan)
    inspector = engine.create_engine_inspector()
    (args.output / 'engine-inspector.json').write_text(inspector.get_engine_information(trt.LayerInformationFormat.JSON))
    context = engine.create_execution_context()
    tensors = {k: torch.from_numpy(v).cuda().contiguous() for k, v in input_values.items()}
    for key, _ in operations:
        tensors[key + '_actual'] = torch.empty(captured[key + '_expected'].shape, device='cuda')
    for name, tensor in tensors.items():
        context.set_tensor_address(name, tensor.data_ptr())
    if not context.execute_async_v3(torch.cuda.current_stream().cuda_stream):
        raise RuntimeError('Operation probe execution failed')
    torch.cuda.synchronize()
    rows = []
    for key, kind in operations:
        ref = captured[key + '_expected']
        got = tensors[key + '_actual'].cpu().numpy()
        difference = np.abs(got - ref)
        row = {'op': key, 'kind': kind, 'max_abs': float(difference.max()),
               'mean_abs': float(difference.mean()),
               'max_tolerance_ratio': float((difference / (ATOL + RTOL * np.abs(ref))).max()),
               'passed': bool(np.allclose(got, ref, atol=ATOL, rtol=RTOL))}
        rows.append(row)
        print(json.dumps(row), flush=True)
    (args.output / 'report.json').write_text(json.dumps({
        'revision': revision, 'checkpoint_sha256': FLOW_SHA256,
        'frames': args.frames, 'masked': args.masked,
        'plan_sha256': sha256_file(args.component / 'flow.plan'),
        'repeat_max_errors': [float(np.abs(v-actual[0]).max()) for v in actual],
        'whole_model_max_error': float(np.abs(actual[0]-expected).max()),
        'operations': rows,
    }, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()

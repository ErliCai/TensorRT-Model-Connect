# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare attention scaling orders using captured official block inputs."""
import argparse
import json
from pathlib import Path

import numpy as np

from ..checkpoint_mapper import load_flow_weights
from ..config import FlowConfig, ShapeProfile
from ..flow_builder import _Graph
from ..validation.validate_flow_pytorch import ATOL, RTOL, _ieee_fp32_reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    import torch
    import torch.nn.functional as F
    import tensorrt as trt
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from x_transformers.x_transformers import RotaryEmbedding, apply_rotary_pos_emb

    if args.report.exists():
        raise FileExistsError(args.report)
    data = np.load(args.capture)
    if 'input_mask' not in data or not np.all(data['input_mask'] == 1):
        raise ValueError('This probe requires a new unmasked capture from diagnose_flow_ops')
    weights = load_flow_weights(args.model_dir, FlowConfig())
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    net = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    g = _Graph(net, weights, FlowConfig(), ShapeProfile(4, 32, 128))
    inputs, expected, rows = {}, {}, []

    def record(name, actual, reference):
        diff = np.abs(actual-reference)
        row = dict(name=name, passed=bool(np.allclose(actual, reference, atol=ATOL, rtol=RTOL)),
                   max_abs=float(diff.max()), mean_abs=float(diff.mean()),
                   ratio=float((diff/(ATOL+RTOL*np.abs(reference))).max()))
        rows.append(row)
        print(json.dumps(row), flush=True)

    for i in (0, 2, 18, 19, 20, 21):
        key = f'transformer_blocks.{i}.attn'
        arr = data[key + '_input']
        frames = arr.shape[1]
        ref = data[key + '_expected']
        phases = np.repeat(np.arange(frames, dtype=np.float32)[:, None] * weights['rotary_embed.inv_freq'][None], 2, axis=-1)
        cos, sin = g.const(np.cos(phases)[None]), g.const(np.sin(phases)[None])
        inp = net.add_input(key, trt.float32, arr.shape)
        inputs[key] = arr
        q = g.rotary(g.linear(inp, key+'.to_q'), cos, sin)
        k = g.rotary(g.linear(inp, key+'.to_k'), cos, sin)
        v = g.linear(inp, key+'.to_v')
        q,k,v = [g.transpose(g.reshape(a, (2, frames, 16, 64)), (0,2,1,3)) for a in (q,k,v)]
        for variant in ('split', 'q_only', 'post', 'center_key'):
            qa,ka = q,k
            if variant == 'split':
                qa,ka = g.scale(q,64**-.25),g.scale(k,64**-.25)
            elif variant in ('q_only','center_key'):
                qa = g.scale(q,.125)
                if variant == 'center_key':
                    mean = net.add_reduce(k,trt.ReduceOperation.AVG,1<<2,True).get_output(0)
                    ka = g.ew(k,mean,'SUB')
            scores = net.add_matrix_multiply(qa,trt.MatrixOperation.NONE,ka,trt.MatrixOperation.TRANSPOSE).get_output(0)
            if variant == 'post':
                scores = g.scale(scores,.125)
            softmax = net.add_softmax(scores)
            softmax.axes = 1<<3
            out = net.add_matrix_multiply(softmax.get_output(0),trt.MatrixOperation.NONE,v,trt.MatrixOperation.NONE).get_output(0)
            out = g.linear(g.reshape(g.transpose(out,(0,2,1,3)),(2,frames,1024)),key+'.to_out.0')
            out.name = f'b{i}_{variant}'
            net.mark_output(out)
            expected[out.name] = ref

        with torch.inference_mode(), _ieee_fp32_reference(torch):
            rope,_ = RotaryEmbedding(64).cuda().forward_from_seq_len(frames)
            q,k,v = [torch.from_numpy(data[key+'.to_'+name+'_expected']).cuda() for name in ('q','k','v')]
            q,k = apply_rotary_pos_emb(q,rope),apply_rotary_pos_emb(k,rope)
            q,k,v = [a.reshape(2,frames,16,64).transpose(1,2) for a in (q,k,v)]
            w = torch.from_numpy(weights[key+'.to_out.0.weight']).cuda()
            b = torch.from_numpy(weights[key+'.to_out.0.bias']).cuda()
            mask = torch.ones((2,1,frames,frames),dtype=torch.bool,device='cuda')
            for variant in ('torch_auto','torch_math','torch_double'):
                if variant == 'torch_auto':
                    out = F.scaled_dot_product_attention(q,k,v,attn_mask=mask)
                elif variant == 'torch_math':
                    with sdpa_kernel(SDPBackend.MATH):
                        out = F.scaled_dot_product_attention(q,k,v,attn_mask=mask)
                else:
                    out = F.scaled_dot_product_attention(q.double(),k.double(),v.double(),attn_mask=mask).float()
                out = F.linear(out.transpose(1,2).reshape(2,frames,1024),w,b)
                record(f'b{i}_{variant}',out.cpu().numpy(),ref)
    config = builder.create_builder_config()
    config.builder_optimization_level = 4
    config.clear_flag(trt.BuilderFlag.TF32)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,256*1024*1024)
    plan = builder.build_serialized_network(net,config)
    if plan is None:
        raise RuntimeError('Probe build failed')
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(plan)
    context = engine.create_execution_context()
    tensors = {k:torch.from_numpy(v).cuda().contiguous() for k,v in inputs.items()}
    tensors.update({k:torch.empty(v.shape,device='cuda') for k,v in expected.items()})
    for name,tensor in tensors.items():
        context.set_tensor_address(name,tensor.data_ptr())
    if not context.execute_async_v3(torch.cuda.current_stream().cuda_stream):
        raise RuntimeError('Attention probe execution failed')
    torch.cuda.synchronize()
    for name,ref in expected.items():
        record(name,tensors[name].cpu().numpy(),ref)
    with args.report.open('x', encoding='utf-8') as handle:
        json.dump(rows, handle, indent=2)


if __name__ == '__main__':
    main()

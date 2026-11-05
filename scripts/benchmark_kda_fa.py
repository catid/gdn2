#!/usr/bin/env python3
"""Matched token-mixer benchmark for GDN-2, KDA, and flash attention.

This is a reproducible layer-level baseline. It includes the token mixer,
backward, and fused AdamW step. GDN-2 uses this checkout; KDA uses the pinned
Flash Linear Attention implementation; FA uses PyTorch's causal SDPA backend
(which dispatches to the Blackwell flash-attention kernel when available).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import types
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import the layer without importing the training-only lit_gpt package
# initializer (which brings in Lightning and dataset dependencies).
_repo_root = Path(__file__).resolve().parents[1]
_lit_pkg = types.ModuleType("lit_gpt")
_lit_pkg.__path__ = [str(_repo_root / "lit_gpt")]
sys.modules.setdefault("lit_gpt", _lit_pkg)
from lit_gpt.gdn2 import GatedDeltaNet2


class FlashAttentionBaseline(nn.Module):
    """Simple causal FA token mixer using PyTorch SDPA."""

    def __init__(self, hidden_size: int, num_heads: int, head_dim: int) -> None:
        super().__init__()
        if hidden_size != num_heads * head_dim:
            raise ValueError("FA baseline requires hidden_size == num_heads * head_dim")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.norm = nn.RMSNorm(hidden_size, eps=1e-5)
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.norm(x)
        q, k, v = self.qkv(z).view(
            z.shape[0], z.shape[1], 3, self.num_heads, self.head_dim
        ).unbind(2)
        y = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True
        )
        return x + self.o_proj(y.transpose(1, 2).reshape_as(x))


def _replace_linears(module: nn.Module) -> None:
    from transformer_engine.pytorch import Linear as TELinear

    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            replacement = TELinear(
                child.in_features,
                child.out_features,
                bias=child.bias is not None,
                params_dtype=torch.bfloat16,
                device="cuda",
            )
            with torch.no_grad():
                replacement.weight.copy_(child.weight.to(torch.bfloat16))
                if child.bias is not None:
                    replacement.bias.copy_(child.bias.to(torch.bfloat16))
            setattr(module, name, replacement)
        else:
            _replace_linears(child)


def _unwrap(value):
    return value[0] if isinstance(value, tuple) else value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", choices=("gdn2", "kda", "fa"), required=True)
    parser.add_argument("--backend", choices=("triton", "cudnn"), default="triton")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq", type=int, default=4096)
    parser.add_argument("--hidden", type=int, default=3072)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--heads", type=int, default=48)
    parser.add_argument("--expand-v", type=float, default=2.0)
    parser.add_argument("--fp8", action="store_true")
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.hidden != args.heads * args.head_dim:
        raise ValueError("hidden must equal heads * head-dim")
    torch.manual_seed(1234)

    if args.layer == "gdn2":
        layer = GatedDeltaNet2(
            hidden_size=args.hidden,
            head_dim=args.head_dim,
            num_heads=args.heads,
            expand_v=args.expand_v,
            mode="chunk",
            use_short_conv=False,
            backend=args.backend,
            fused_projections=True,
        )
    elif args.layer == "kda":
        from fla.layers.kda import KimiDeltaAttention

        layer = KimiDeltaAttention(
            hidden_size=args.hidden,
            head_dim=args.head_dim,
            num_heads=args.heads,
            expand_v=args.expand_v,
            mode="chunk",
            use_short_conv=False,
        )
    else:
        layer = FlashAttentionBaseline(args.hidden, args.heads, args.head_dim)

    layer = layer.cuda().to(torch.bfloat16).train()
    recipe = None
    fp8_autocast = None
    if args.fp8:
        from transformer_engine.common.recipe import DelayedScaling, Format
        from transformer_engine.pytorch import fp8_autocast as _fp8_autocast

        _replace_linears(layer)
        recipe = DelayedScaling(fp8_format=Format.HYBRID, amax_history_len=16, amax_compute_algo="max")
        fp8_autocast = _fp8_autocast

    x = torch.randn(args.batch, args.seq, args.hidden, device="cuda", dtype=torch.bfloat16)
    optimizer = torch.optim.AdamW(
        layer.parameters(), lr=1e-4, betas=(0.9, 0.95), weight_decay=0.1,
        fused=True, capturable=args.graph,
    )

    def step():
        optimizer.zero_grad(set_to_none=False if args.graph else True)
        if fp8_autocast is None:
            y = _unwrap(layer(x))
            loss = y.float().square().mean()
            loss.backward()
        else:
            with fp8_autocast(enabled=True, fp8_recipe=recipe):
                y = _unwrap(layer(x))
                loss = y.float().square().mean()
                loss.backward()
        optimizer.step()
        return loss

    for _ in range(args.warmup):
        step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    timings = []
    if args.graph:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            loss = step()
        torch.cuda.synchronize()
        for _ in range(args.steps):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            graph.replay()
            end.record()
            end.synchronize()
            timings.append(start.elapsed_time(end))
    else:
        for _ in range(args.steps):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            loss = step()
            end.record()
            end.synchronize()
            timings.append(start.elapsed_time(end))

    mean_ms = statistics.fmean(timings)
    print(json.dumps({
        "layer": args.layer,
        "backend": args.backend if args.layer == "gdn2" else None,
        "batch": args.batch,
        "sequence_length": args.seq,
        "hidden_size": args.hidden,
        "head_dim": args.head_dim,
        "num_heads": args.heads,
        "expand_v": args.expand_v if args.layer != "fa" else 1.0,
        "parameters": sum(p.numel() for p in layer.parameters()),
        "fp8": args.fp8,
        "cuda_graph": args.graph,
        "mean_step_ms": mean_ms,
        "median_step_ms": statistics.median(timings),
        "tokens_per_second": args.batch * args.seq / (mean_ms / 1000),
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "loss": float(loss.detach()),
        "step_times_ms": timings,
    }, sort_keys=True))


if __name__ == "__main__":
    main()

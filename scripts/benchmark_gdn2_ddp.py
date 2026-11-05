#!/usr/bin/env python3
"""DDP/CUDA-Graph benchmark for the fixed-shape GDN-2 training step.

Launch with ``torchrun --standalone --nproc_per_node=<N>``.  Each rank owns a
GDN-2 layer and a local microbatch; DDP performs the gradient all-reduce.  The
script is intentionally a layer benchmark so it can be run before committing
to a full-model batch size. For an apples-to-apples 8xB300 measurement, run
it on eight identical GPUs; rank zero reports the slowest rank's global token
rate.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Avoid importing lit_gpt.__init__, which pulls in optional training-only
# dependencies such as Lightning.
_repo_root = Path(__file__).resolve().parents[1]
_lit_pkg = types.ModuleType("lit_gpt")
_lit_pkg.__path__ = [str(_repo_root / "lit_gpt")]
sys.modules.setdefault("lit_gpt", _lit_pkg)
from lit_gpt.gdn2 import GatedDeltaNet2


def _fp8_replace_linears(module: torch.nn.Module) -> None:
    from transformer_engine.pytorch import Linear as TELinear

    for name, child in list(module.named_children()):
        if isinstance(child, torch.nn.Linear):
            replacement = TELinear(
                child.in_features,
                child.out_features,
                bias=child.bias is not None,
                params_dtype=torch.bfloat16,
                device=torch.cuda.current_device(),
            )
            with torch.no_grad():
                replacement.weight.copy_(child.weight.to(torch.bfloat16))
                if child.bias is not None:
                    replacement.bias.copy_(child.bias.to(torch.bfloat16))
            setattr(module, name, replacement)
        else:
            _fp8_replace_linears(child)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("triton", "cudnn"), default="triton")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--fp8", action="store_true")
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--fused-projections", dest="fused_projections", action="store_true", default=True)
    parser.add_argument("--no-fused-projections", dest="fused_projections", action="store_false")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    launched_distributed = "RANK" in os.environ
    if launched_distributed:
        dist.init_process_group("nccl")
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    # Use identical initial weights across ranks, as DDP requires.  Inputs are
    # seeded per rank below so each process still does distinct work.
    torch.manual_seed(1234)

    layer = GatedDeltaNet2(
        hidden_size=3072,
        head_dim=64,
        num_heads=48,
        expand_v=2,
        mode="chunk",
        use_short_conv=False,
        backend=args.backend,
        fused_projections=args.fused_projections,
    ).to(device=device, dtype=torch.bfloat16).train()
    recipe = None
    fp8_autocast = None
    if args.fp8:
        from transformer_engine.common.recipe import DelayedScaling, Format
        from transformer_engine.pytorch import fp8_autocast as _fp8_autocast

        _fp8_replace_linears(layer)
        recipe = DelayedScaling(fp8_format=Format.HYBRID, amax_history_len=16, amax_compute_algo="max")
        fp8_autocast = _fp8_autocast
    # Capturing DDP itself is not portable: DDP's Python logger and reducer
    # setup perform host-side work that CUDA Graph capture rejects.  In graph
    # mode use a graph-safe manual gradient all-reduce instead.  This preserves
    # DDP semantics while keeping the model and optimizer graph-capturable.
    manual_sync = launched_distributed and args.graph
    if launched_distributed and not manual_sync:
        layer = DDP(layer, device_ids=[local_rank], gradient_as_bucket_view=True, static_graph=True)

    torch.manual_seed(4321 + rank)
    x = torch.randn(args.batch, args.seq, 3072, device=device, dtype=torch.bfloat16)
    optimizer = torch.optim.AdamW(
        layer.parameters(), lr=1e-4, betas=(0.9, 0.95), weight_decay=0.1,
        fused=True, capturable=args.graph,
    )

    def step():
        optimizer.zero_grad(set_to_none=False if args.graph else True)
        if fp8_autocast is not None:
            with fp8_autocast(enabled=True, fp8_recipe=recipe):
                y = layer(x)[0]
                loss = y.float().square().mean()
                loss.backward()
        else:
            y = layer(x)[0]
            loss = y.float().square().mean()
            loss.backward()
        optimizer.step()
        return loss

    def compute_step():
        """Graph-safe local forward/backward, without distributed collectives."""
        optimizer.zero_grad(set_to_none=False)
        if fp8_autocast is not None:
            with fp8_autocast(enabled=True, fp8_recipe=recipe):
                y = layer(x)[0]
                loss = y.float().square().mean()
                loss.backward()
        else:
            y = layer(x)[0]
            loss = y.float().square().mean()
            loss.backward()
        return loss

    sync_groups = {}

    def sync_grads():
        nonlocal sync_groups
        if not manual_sync:
            return
        grads = [parameter.grad for parameter in layer.parameters() if parameter.grad is not None]
        if not grads:
            return
        # One coalesced NCCL collective avoids a launch per parameter.  The
        # graph path intentionally performs this outside the captured local
        # compute graph, so the flat buffer can be reused by the next step.
        grouped = {}
        for grad in grads:
            grouped.setdefault(grad.dtype, []).append(grad)
        for dtype, dtype_grads in grouped.items():
            flat, views = sync_groups.get(dtype, (None, None))
            if flat is None:
                flat = torch.empty(
                    sum(grad.numel() for grad in dtype_grads), dtype=dtype, device=device
                )
                views = list(flat.split([grad.numel() for grad in dtype_grads]))
                sync_groups[dtype] = (flat, views)
            for view, grad in zip(views, dtype_grads):
                view.copy_(grad.reshape(-1))
            dist.all_reduce(flat, op=dist.ReduceOp.SUM)
            flat.div_(world)
            for grad, view in zip(dtype_grads, views):
                grad.copy_(view.reshape_as(grad))

    if manual_sync:
        for _ in range(args.warmup):
            compute_step()
            sync_grads()
            optimizer.step()
    else:
        for _ in range(args.warmup):
            step()
    torch.cuda.synchronize(device)
    if launched_distributed:
        dist.barrier()
    timings = []
    if args.graph and manual_sync:
        # Capture local compute first.  The NCCL all-reduces happen between
        # the compute and optimizer graphs, avoiding DDP reducer capture.
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            loss = compute_step()
        torch.cuda.synchronize(device)
        for _ in range(args.steps):
            start = time.perf_counter()
            graph.replay()
            sync_grads()
            optimizer.step()
            torch.cuda.synchronize(device)
            dist.barrier()
            timings.append((time.perf_counter() - start) * 1000.0)
    elif args.graph:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            loss = step()
        torch.cuda.synchronize(device)
        for _ in range(args.steps):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            graph.replay()
            end.record()
            end.synchronize()
            if launched_distributed:
                dist.barrier()
            timings.append(start.elapsed_time(end))
    else:
        for _ in range(args.steps):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            loss = step()
            end.record()
            end.synchronize()
            if launched_distributed:
                dist.barrier()
            timings.append(start.elapsed_time(end))

    mean_ms = statistics.fmean(timings)
    if launched_distributed:
        # The step is paced by the slowest rank.  Report the max rank time,
        # rather than rank zero's local event duration.
        step_ms = torch.tensor(mean_ms, device=device, dtype=torch.float64)
        dist.all_reduce(step_ms, op=dist.ReduceOp.MAX)
        mean_ms = float(step_ms.item())
    if rank == 0:
        print(json.dumps({
            "backend": args.backend,
            "batch_per_rank": args.batch,
            "sequence_length": args.seq,
            "world_size": world,
            "fp8": args.fp8,
            "cuda_graph": args.graph,
            "fused_projections": args.fused_projections,
            "mean_step_ms": mean_ms,
            "global_tokens_per_second": world * args.batch * args.seq / (mean_ms / 1000),
            "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20,
            "loss": float(loss.detach()),
            "step_times_ms": timings,
        }, sort_keys=True), flush=True)
    if launched_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

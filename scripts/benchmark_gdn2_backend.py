#!/usr/bin/env python3
"""Small fixed-shape GDN-2 forward/backward benchmark.

This measures the recurrent layer (including its projections, backward, and
fused AdamW), rather than a forward-only kernel microbenchmark.  The native
backend requires cuDNN 1.32+ on an SM100/SM103 GPU.
"""

import argparse
import json
import statistics
import sys
import types
from pathlib import Path

import torch
from torch import nn

# Avoid importing lit_gpt.__init__, which pulls in optional training-only
# dependencies such as Lightning.
_repo_root = Path(__file__).resolve().parents[1]
_lit_pkg = types.ModuleType("lit_gpt")
_lit_pkg.__path__ = [str(_repo_root / "lit_gpt")]
sys.modules.setdefault("lit_gpt", _lit_pkg)
from lit_gpt.gdn2 import GatedDeltaNet2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("triton", "cudnn"), default="triton")
    parser.add_argument("--fused-projections", action="store_true")
    parser.add_argument(
        "--disable-recompute",
        action="store_true",
        help="Keep chunk intermediates for backward; uses more VRAM but can reduce step time",
    )
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--fp8", action="store_true", help="Use TransformerEngine FP8 Linear projections")
    parser.add_argument(
        "--fp8-scaling",
        choices=("delayed", "current", "block"),
        default="delayed",
        help="TransformerEngine FP8 scaling recipe (default: delayed)",
    )
    parser.add_argument("--graph", action="store_true", help="Capture the fixed-shape training step in a CUDA graph")
    parser.add_argument(
        "--fastest",
        action="store_true",
        help="Select the recommended B300 path: FROST + packed projections + TE FP8 + graph",
    )
    args = parser.parse_args()

    if args.fastest:
        args.backend = "cudnn"
        args.fused_projections = True
        args.fp8 = True
        args.graph = True

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")
    if args.backend == "cudnn":
        # Importing the Python op is not enough to establish that the native
        # FROST engine is installed.  Older cuDNN frontend packages expose the
        # symbol but fail only when the execution plan is created, producing a
        # misleading backend error several seconds into the benchmark.
        try:
            import cudnn
            version_text = str(getattr(cudnn, "__version__", "0.0"))
            version = tuple(int(part) for part in version_text.split(".")[:2])
        except Exception as exc:
            raise RuntimeError(
                "backend='cudnn' requires the cuDNN Python package with the "
                "native GDN-2/FROST engine (cuDNN frontend >= 1.32); "
                "use --backend triton in this environment"
            ) from exc
        if version < (1, 32):
            raise RuntimeError(
                f"backend='cudnn' requires cuDNN frontend >= 1.32 for native "
                f"GDN-2/FROST, but found {version_text}; use --backend triton "
                "or a newer cuDNN container"
            )
    torch.manual_seed(1234)
    hidden = 3072
    layer = GatedDeltaNet2(
        hidden_size=hidden,
        expand_v=2,
        head_dim=64,
        num_heads=hidden // 64,
        mode="chunk",
        use_short_conv=False,
        backend=args.backend,
        fused_projections=args.fused_projections,
        disable_recompute=args.disable_recompute,
    ).cuda().to(torch.bfloat16).train()

    recipe = None
    if args.fp8:
        from transformer_engine.common.recipe import DelayedScaling, Float8BlockScaling, Float8CurrentScaling, Format
        from transformer_engine.pytorch import fp8_autocast, Linear as TELinear

        def replace_linears(module: nn.Module) -> None:
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
                    replace_linears(child)

        replace_linears(layer)
        if args.fp8_scaling == "current":
            recipe = Float8CurrentScaling(fp8_format=Format.HYBRID)
        elif args.fp8_scaling == "block":
            recipe = Float8BlockScaling(fp8_format=Format.E4M3)
        else:
            recipe = DelayedScaling(fp8_format=Format.HYBRID, amax_history_len=16, amax_compute_algo="max")
    x = torch.randn(args.batch, args.seq, hidden, device="cuda", dtype=torch.bfloat16)
    optimizer = torch.optim.AdamW(
        layer.parameters(), lr=1e-4, betas=(0.9, 0.95), weight_decay=0.1,
        fused=True, capturable=args.graph,
    )

    def step():
        optimizer.zero_grad(set_to_none=False if args.graph else True)
        if args.fp8:
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
    tokens = args.batch * args.seq
    print(json.dumps({
        "backend": args.backend,
        "fused_projections": args.fused_projections,
        "disable_recompute": args.disable_recompute,
        "fp8": args.fp8,
        "fp8_scaling": args.fp8_scaling if args.fp8 else None,
        "cuda_graph": args.graph,
        "fastest_path": args.backend == "cudnn" and args.fused_projections and args.fp8 and args.graph,
        "batch": args.batch,
        "sequence_length": args.seq,
        "mean_step_ms": mean_ms,
        "median_step_ms": statistics.median(timings),
        "tokens_per_second": tokens / (mean_ms / 1000),
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "loss": float(loss.detach()),
        "step_times_ms": timings,
    }, sort_keys=True))


if __name__ == "__main__":
    main()

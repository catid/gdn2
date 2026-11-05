# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

r"""
GDN-2 (Gated DeltaNet 2) token-mixing layer.

This module defines `GatedDeltaNet2`, the `nn.Module` that wraps the GDN-2
recurrence into a drop-in token mixer for a Transformer-style block. It
handles projections, short convolutions, gate construction, kernel dispatch,
caching for incremental decoding, and the gated output normalization.

The recurrence itself is implemented by the Triton kernels in `gdn2_ops`;
this layer only prepares their inputs and consumes their outputs. Two kernels
are dispatched between automatically: the chunkwise kernel `chunk_gdn2` for
training and long sequences, and the token-by-token `fused_recurrent_gdn2`
for short-sequence decoding.

GDN-2 replaces the scalar write-strength gate of the gated delta rule with two
independent channel-wise gates: an erase gate `b` on the key axis and a write
gate `w` on the value axis. See the kernel modules for the recurrence itself.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Literal

import torch
import torch.nn as nn
from einops import rearrange, repeat
from torch.nn import functional as F

from fla.layers.utils import get_layer_cache, get_unpad_data, index_first_axis, pad_input, update_layer_cache
from fla.modules import FusedRMSNormSwishGate, ShortConvolution
from fla.modules.l2norm import l2norm_bwd, l2norm_fwd

from .gdn2_ops.chunk_gdn2 import _GDN2BetaGuard, chunk_gdn2
from .gdn2_ops.fused_recurrent_gdn2 import fused_recurrent_gdn2

# cuDNN 1.32+ ships a native SM100/SM103 GDN-2 engine (FROST).  Keep it an
# optional import so the Triton implementation remains the default and the
# package is still importable on machines without the cuDNN linear-attention
# extension.  The native path is selected explicitly with ``backend='cudnn'``.
try:
    from cudnn.linear_attention.ops import gated_delta_net_v2 as _cudnn_gdn2
except Exception:  # optional dependency, and some older cuDNN builds lack it
    _cudnn_gdn2 = None


class _GDN2L2Norm(torch.autograd.Function):
    """Autograd bridge for FLA's fused L2 normalization kernel."""

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        y, rstd = l2norm_fwd(x)
        ctx.save_for_backward(y, rstd)
        return y

    @staticmethod
    def backward(ctx, dy: torch.Tensor) -> tuple[torch.Tensor]:
        y, rstd = ctx.saved_tensors
        return (l2norm_bwd(y, rstd, dy),)

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack

    from fla.models.utils import Cache


class GatedDeltaNet2(nn.Module):
    """
    Gated DeltaNet 2 (GDN-2) layer implementation.

    GDN-2 extends KDA's scalar-beta erase gate to channel-wise erase and write
    gates:

        S_t = (I - k_t (b_t ⊙ k_t)^T) Diag(exp(g_t)) S_{t-1}
              + k_t (w_t ⊙ v_t)^T

    Here b_t ∈ R^{d_k} is the channel-wise erase gate (replacing KDA's scalar
    beta_t) and w_t ∈ R^{d_v} is the channel-wise write gate (new in GDN-2).
    Setting b_t = beta_t · 1 and w_t = beta_t · 1 recovers KDA exactly.

    Args:
        hidden_size (int, Optional):
            The hidden size of the input. Default: 2048.
        expand_v (float, Optional):
            The expansion ratio for the value dimension. Default: 1.0.
        head_dim (int, Optional):
            The dimension of each head. Default: 128.
        num_heads (int, Optional):
            The number of heads. Default: 16.
        num_v_heads (int, Optional):
            The number of heads for the value projection, equal to `num_heads` if `None`.
            GVA (Grouped Value Attention) is applied if `num_v_heads` > `num_heads`. Default: `None`.
        mode (str, Optional):
            Which GDN-2 kernel to use. Available: `chunk` (training + long inference)
            and `fused_recurrent` (token-by-token decode, inference only).
            The layer automatically falls back to `fused_recurrent` for short
            inference sequences (q_len <= 64); otherwise `self.mode` is used.
            Default: `chunk`.
        use_short_conv (bool, Optional):
            Whether to use short convolutions. Default: `True`.
        allow_neg_eigval (bool, Optional):
            Allow negative eigenvalues. Default: `False`. If set to `True`, the
            erase gate `b` will be multiplied by 2.
            See reference:
            [Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues](https://arxiv.org/abs/2411.12537)
        conv_size (int, Optional):
            The kernel size of the short convolution, only used when `use_short_conv` is `True`. Default: 4.
        conv_bias (bool, Optional):
            Whether to use bias in the short convolution, only used when `use_short_conv` is `True`. Default: `False`.
        disable_recompute (bool, Optional):
            Keep chunkwise intermediates for backward, trading activation
            memory for less recomputation on the Triton chunk path. Default:
            `False`.
        layer_idx (int, Optional):
            The index of the layer. Default: None.
        norm_eps (float, Optional):
            The epsilon value for the normalization layer. Default: 1e-5.
    """

    def __init__(
        self,
        hidden_size: int = 2048,
        expand_v: float = 1,
        head_dim: int = 128,
        num_heads: int = 16,
        num_v_heads: int = None,
        mode: Literal["chunk", "fused_recurrent"] = "chunk",
        use_short_conv: bool = True,
        allow_neg_eigval: bool = False,
        conv_size: int = 4,
        conv_bias: bool = False,
        fused_projections: bool = False,
        backend: Literal["triton", "cudnn"] = "triton",
        beta_guard: bool = True,
        disable_recompute: bool = False,
        layer_idx: int = None,
        norm_eps: float = 1e-5,
        **kwargs,
    ) -> GatedDeltaNet2:
        super().__init__()

        self.mode = mode
        self.allow_neg_eigval = allow_neg_eigval
        self.hidden_size = hidden_size
        self.expand_v = expand_v

        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.conv_bias = conv_bias
        self.fused_projections = bool(fused_projections)
        if backend not in ("triton", "cudnn"):
            raise ValueError(f"Unsupported GDN-2 backend `{backend}`.")
        if backend == "cudnn" and _cudnn_gdn2 is None:
            raise ImportError(
                "backend='cudnn' requires cuDNN 1.32+ with the linear-attention GDN-2 op"
            )
        self.backend = backend
        # Keep chunk intermediates for backward instead of recomputing them.
        # This trades activation memory for lower step latency and is useful
        # for fixed-shape CUDA-graph training when the memory budget permits.
        self.disable_recompute = bool(disable_recompute)
        # Cache dense-batch cu_seqlens vectors used by the native THD adapter.
        self._cudnn_cu_seqlens_cache: dict[tuple[int, int, str], torch.Tensor] = {}
        self.beta_guard = beta_guard

        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_v_heads = num_v_heads if num_v_heads is not None else num_heads

        self.head_k_dim = head_dim
        self.head_v_dim = int(self.head_dim * self.expand_v)
        self.key_dim = int(self.num_heads * self.head_k_dim)
        self.value_dim = int(self.num_v_heads * self.head_v_dim)
        self.layer_idx = layer_idx

        if self.backend == "cudnn" and (
            self.head_k_dim not in (64, 128) or self.head_v_dim not in (64, 128)
        ):
            raise ValueError(
                "backend='cudnn' currently supports head key/value dimensions "
                f"64 or 128, got K={self.head_k_dim}, V={self.head_v_dim}."
            )

        # Consistency check: Ensure expand_v produces integer values
        if not math.isclose(self.num_v_heads * self.head_dim * expand_v, self.value_dim, rel_tol=1e-5):
            raise ValueError(
                f"expand_v={expand_v} does not produce an integer value when multiplied by key_dim={self.key_dim}. "
                f"Resulting value_dim would be {self.num_v_heads * self.head_dim * expand_v}, which is invalid for nn.Linear.",
            )
        if self.num_v_heads > self.num_heads and self.num_v_heads % self.num_heads != 0:
            raise ValueError(
                f"num_v_heads={self.num_v_heads} must be divisible by num_heads={self.num_heads}.",
            )

        if not math.isclose(head_dim * expand_v, self.head_v_dim, rel_tol=1e-5):
            raise ValueError(
                f"expand_v={expand_v} does not produce an integer value when multiplied by head_dim={head_dim}. "
                f"Resulting head_v_dim would be {head_dim * expand_v}, which is invalid for FusedRMSNormSwishGate.",
            )
        assert mode in ["chunk", "fused_recurrent"], f"Not supported mode `{mode}`."

        # Query / key / value projections.  The optional packed projection
        # emits q, k, v, erase, write, and the low-rank f/g inputs in one GEMM;
        # keeping it opt-in
        # preserves the original state-dict/module layout by default.
        if self.fused_projections:
            self.qkvbwfg_proj = nn.Linear(
                hidden_size,
                2 * self.key_dim + 2 * self.value_dim + self.key_dim + 2 * self.head_v_dim,
                bias=False,
            )
        else:
            self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
            self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
            self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)

        # Optional depthwise short convolutions on q, k, v. These give the
        # model a small local receptive field before the recurrence and are
        # standard in the gated delta rule family.
        if use_short_conv:
            self.q_conv1d = ShortConvolution(
                hidden_size=self.key_dim,
                kernel_size=conv_size,
                bias=conv_bias,
                activation="silu",
            )
            self.k_conv1d = ShortConvolution(
                hidden_size=self.key_dim,
                kernel_size=conv_size,
                bias=conv_bias,
                activation="silu",
            )
            self.v_conv1d = ShortConvolution(
                hidden_size=self.value_dim,
                kernel_size=conv_size,
                bias=conv_bias,
                activation="silu",
            )

        # Decay-gate projection. Produces the pre-activation that, combined
        # with A_log and dt_bias below, yields the channel-wise log-decay g.
        if self.fused_projections:
            # The f/g input projections are emitted by qkvbwfg_proj above.
            self.f_proj = nn.Linear(self.head_v_dim, self.key_dim, bias=False)
        else:
            self.f_proj = nn.Sequential(
                nn.Linear(hidden_size, self.head_v_dim, bias=False),
                nn.Linear(self.head_v_dim, self.key_dim, bias=False),
            )

        # GDN-2 channel-wise gates. b_proj produces the erase gate on the key
        # axis; w_proj produces the write gate on the value axis. Together
        # they replace the single scalar write-strength gate of KDA.
        if not self.fused_projections:
            self.b_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
            self.w_proj = nn.Linear(hidden_size, self.value_dim, bias=False)

        # Decay-gate parameters. A_log is a per-head log-rate; dt_bias is a
        # per-channel bias initialized so the softplus step-size starts in a
        # small range. Both are excluded from weight decay.
        self.A_log = nn.Parameter(torch.log(torch.empty(self.num_heads, dtype=torch.float32).uniform_(1, 16)))
        self.A_log._no_weight_decay = True
        dt = torch.exp(
            torch.rand(self.key_dim, dtype=torch.float32) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
        ).clamp(min=1e-4)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        self.dt_bias._no_weight_decay = True

        # Output path: SiLU-gated RMS norm followed by the output projection.
        if self.fused_projections:
            self.g_proj = nn.Sequential(
                nn.Identity(),
                nn.Linear(self.head_v_dim, self.value_dim, bias=True),
            )
        else:
            self.g_proj = nn.Sequential(
                nn.Linear(hidden_size, self.head_v_dim, bias=False),
                nn.Linear(self.head_v_dim, self.value_dim, bias=True),
            )
        self.o_norm = FusedRMSNormSwishGate(self.head_v_dim, eps=norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)
        self.apply(self._initialize_weights)

    def _initialize_weights(self, module: nn.Module):
        """Xavier-uniform init for all linear layers, applied via `self.apply`.

        The `_is_hf_initialized` guard makes this idempotent so that weights
        loaded by HuggingFace `from_pretrained` are not overwritten.
        """
        if getattr(module, "_is_hf_initialized", False):
            return
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=2 ** -2.5)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        module._is_hf_initialized = True

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """Migrate projection keys between packed and legacy checkpoints.

        Packed mode is an opt-in layout optimization, so checkpoints produced
        by the original five-projection implementation must remain loadable.
        The reverse conversion also lets a packed checkpoint be inspected or
        fine-tuned with the portable legacy path.
        """
        def take(key):
            return state_dict.pop(prefix + key, None)

        def put(key, value):
            if value is not None:
                state_dict[prefix + key] = value

        qkvbw_size = 2 * self.key_dim + 2 * self.value_dim + self.key_dim
        fg_size = 2 * self.head_v_dim
        if self.fused_projections:
            if prefix + "qkvbwfg_proj.weight" not in state_dict:
                packed = take("qkvbw_proj.weight")
                packed_fg = take("fg_proj.weight")
                if packed is not None and packed_fg is not None:
                    put("qkvbwfg_proj.weight", torch.cat((packed, packed_fg), dim=0))
                else:
                    parts = [take(k) for k in (
                        "q_proj.weight", "k_proj.weight", "v_proj.weight",
                        "b_proj.weight", "w_proj.weight",
                    )]
                    fg_parts = [take(k) for k in ("f_proj.0.weight", "g_proj.0.weight")]
                    if all(p is not None for p in parts) and all(p is not None for p in fg_parts):
                        put("qkvbwfg_proj.weight", torch.cat((*parts, *fg_parts), dim=0))
            old_f2 = take("f_proj.1.weight")
            if old_f2 is not None:
                put("f_proj.weight", old_f2)
        else:
            packed_combined = take("qkvbwfg_proj.weight")
            if packed_combined is not None:
                packed, packed_fg = torch.split(packed_combined, (qkvbw_size, fg_size), dim=0)
                sizes = [self.key_dim, self.key_dim, self.value_dim, self.key_dim, self.value_dim]
                for key, value in zip(("q_proj.weight", "k_proj.weight", "v_proj.weight", "b_proj.weight", "w_proj.weight"), torch.split(packed, sizes, dim=0)):
                    put(key, value)
                for key, value in zip(("f_proj.0.weight", "g_proj.0.weight"), torch.split(packed_fg, self.head_v_dim, dim=0)):
                    put(key, value)
            packed = take("qkvbw_proj.weight")
            if packed is not None:
                sizes = [self.key_dim, self.key_dim, self.value_dim, self.key_dim, self.value_dim]
                for key, value in zip(("q_proj.weight", "k_proj.weight", "v_proj.weight", "b_proj.weight", "w_proj.weight"), torch.split(packed, sizes, dim=0)):
                    put(key, value)
            packed_fg = take("fg_proj.weight")
            if packed_fg is not None:
                for key, value in zip(("f_proj.0.weight", "g_proj.0.weight"), torch.split(packed_fg, self.head_v_dim, dim=0)):
                    put(key, value)
            f2 = take("f_proj.weight")
            if f2 is not None:
                put("f_proj.1.weight", f2)

        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys,
            unexpected_keys, error_msgs,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        output_attentions: bool | None = False,
        **kwargs: Unpack[dict],
    ) -> tuple[torch.Tensor, torch.Tensor | None, Cache | None]:
        """Run the GDN-2 token mixer.

        Projects the input to q/k/v and the three gates, dispatches to the
        chunkwise or recurrent kernel, updates the incremental-decoding cache,
        and applies the gated output normalization and projection.

        Args:
            hidden_states: input of shape `[B, T, hidden_size]`.
            attention_mask: optional `[B, T]` 0/1 padding mask. When given,
                the batch is unpadded into a single packed sequence and
                repadded on the way out.
            past_key_values: optional cache holding the recurrent state and
                short-convolution state from previous steps.
            use_cache: whether to write the updated state back into the cache.
            output_attentions: unused; kept for interface compatibility.

        Returns:
            A tuple `(o, None, past_key_values)` where `o` has shape
            `[B, T, hidden_size]`. The second element is always `None`
            (GDN-2 has no attention map to return).
        """
        if attention_mask is not None:
            assert len(attention_mask.shape) == 2, (
                "Expected attention_mask as a 0-1 matrix with shape [batch_size, seq_len] "
                "for padding purposes (0 indicating padding). "
                "Arbitrary attention masks of shape [batch_size, seq_len, seq_len] are not allowed."
            )

        batch_size, q_len, _ = hidden_states.shape
        # Short non-training sequences use the lower-latency recurrent kernel;
        # training and long sequences use the chunkwise kernel.
        mode = "fused_recurrent" if (q_len <= 64 and not self.training) else self.mode
        if self.training:
            assert mode == "chunk", "Only chunk mode is supported in training."

        last_state = get_layer_cache(self, past_key_values)

        cu_seqlens = kwargs.get("cu_seqlens")
        if attention_mask is not None:
            indices, cu_seqlens, _ = get_unpad_data(attention_mask[:, -q_len:])
            hidden_states = index_first_axis(rearrange(hidden_states, "b s ... -> (b s) ..."), indices).unsqueeze(0)

        if self.fused_projections:
            qkvbwfg = self.qkvbwfg_proj(hidden_states)
            qkvbw, fg = torch.split(
                qkvbwfg,
                (2 * self.key_dim + 2 * self.value_dim + self.key_dim, 2 * self.head_v_dim),
                dim=-1,
            )
            q_in, k_in, v_in, b_in, w_in = torch.split(
                qkvbw,
                (self.key_dim, self.key_dim, self.value_dim, self.key_dim, self.value_dim),
                dim=-1,
            )
            f_in, g_out_in = torch.split(fg, self.head_v_dim, dim=-1)
        else:
            q_in = self.q_proj(hidden_states)
            k_in = self.k_proj(hidden_states)
            v_in = self.v_proj(hidden_states)
            f_in = g_out_in = None

        if self.use_short_conv:
            conv_state_q, conv_state_k, conv_state_v = None, None, None
            if last_state is not None:
                conv_state_q, conv_state_k, conv_state_v = last_state["conv_state"]
            q, conv_state_q = self.q_conv1d(
                x=q_in,
                cache=conv_state_q,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
            k, conv_state_k = self.k_conv1d(
                x=k_in,
                cache=conv_state_k,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
            v, conv_state_v = self.v_conv1d(
                x=v_in,
                cache=conv_state_v,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
        else:
            q = F.silu(q_in)
            k = F.silu(k_in)
            v = F.silu(v_in)

        # Channel-wise log-decay, computed in fp32 for numerical stability of
        # the downstream cumulative sum. A_log is per-head and broadcast over
        # the head's key channels; dt_bias is per-channel.
        f_out = self.f_proj(f_in) if self.fused_projections else self.f_proj(hidden_states)
        # Keep the decay in its natural [B, T, H, K] layout while applying
        # the per-head rate.  This avoids materializing a flattened
        # ``repeat_interleave(A_log)`` vector and then immediately reshaping
        # it again.  Besides saving a small allocation, the head-wise form is
        # friendlier to the packed/FROST path because the broadcast follows
        # the recurrence's state layout.
        g = rearrange(
            F.softplus(f_out.float() + self.dt_bias),
            "... (h d) -> ... h d",
            h=self.num_heads,
            d=self.head_k_dim,
        )
        g = -self.A_log.float().exp().view(1, 1, self.num_heads, 1) * g

        # GDN-2 gates, both squashed to [0, 1] by a sigmoid. b is the
        # channel-wise erase gate (key axis); w is the channel-wise write
        # gate (value axis).
        if self.fused_projections:
            b = b_in.sigmoid()
            w = w_in.sigmoid()
            output_gate = self.g_proj[1](g_out_in)
        else:
            b = self.b_proj(hidden_states).sigmoid()
            w = self.w_proj(hidden_states).sigmoid()
            output_gate = self.g_proj(hidden_states)

        # Split the flat projection outputs into per-head tensors. Key-side
        # tensors (q, k, g, b) use head_k_dim; value-side (v, w) use head_v_dim.
        q, k = (rearrange(x, "... (h d) -> ... h d", d=self.head_k_dim) for x in (q, k))
        v = rearrange(v, "... (h d) -> ... h d", d=self.head_v_dim)
        b = rearrange(b, "... (h d) -> ... h d", d=self.head_k_dim)
        w = rearrange(w, "... (h d) -> ... h d", d=self.head_v_dim)

        # Grouped value attention: when there are more value heads than key
        # heads, replicate the key-side tensors across each value-head group.
        if self.num_v_heads > self.num_heads:
            q, k, g, b = (
                repeat(x, "... h d -> ... (h g) d", g=self.num_v_heads // self.num_heads)
                for x in (q, k, g, b)
            )

        # Optionally lift the erase gate from [0, 1] into [0, 2], which allows
        # negative eigenvalues in the state transition (extra state-tracking
        # capacity). The write gate w is left in [0, 1].
        if self.allow_neg_eigval:
            b = b * 2.0

        recurrent_state = last_state["recurrent_state"] if last_state is not None else None
        if mode == "chunk":
            if self.backend == "cudnn":
                # cuDNN's graph API uses packed THD tensors and stores state
                # as [N, H, V, K], whereas FLA's cache uses [N, H, K, V].
                # Keep all projections and nonlinearities in this module so
                # the native recurrence is algebraically identical to the
                # Triton path, including CroissAttn-stable gate construction.
                if cu_seqlens is None:
                    key = (batch_size, q_len, str(q.device))
                    cu_seqlens_native = self._cudnn_cu_seqlens_cache.get(key)
                    if cu_seqlens_native is None:
                        cu_seqlens_native = torch.arange(
                            0,
                            (batch_size + 1) * q_len,
                            q_len,
                            dtype=torch.int32,
                            device=q.device,
                        )
                        self._cudnn_cu_seqlens_cache[key] = cu_seqlens_native
                else:
                    cu_seqlens_native = cu_seqlens.to(dtype=torch.int32)
                # Match the Triton path's normalized-key guard and expose the
                # normalization gradient explicitly because the cuDNN op is
                # called with ``use_qk_l2norm_in_kernel=False`` below.
                q_norm = _GDN2L2Norm.apply(q)
                k_norm = _GDN2L2Norm.apply(k)
                q_native = q_norm.reshape(-1, q_norm.shape[-2], q_norm.shape[-1]).contiguous()
                k_native = k_norm.reshape(-1, k_norm.shape[-2], k_norm.shape[-1]).contiguous()
                v_native = v.reshape(-1, v.shape[-2], v.shape[-1]).contiguous()
                g_native = g.reshape(-1, g.shape[-2], g.shape[-1]).contiguous()
                w_native = w.reshape(-1, w.shape[-2], w.shape[-1]).contiguous()
                state_native = (
                    recurrent_state.transpose(-1, -2).contiguous()
                    if recurrent_state is not None
                    else None
                )
                # FROST has no beta-guard argument.  Apply the same certified
                # projection used by the Triton path, with a straight-through
                # estimator so gradients still flow to the original gate.
                b_native = b
                if self.beta_guard:
                    b_guarded = _GDN2BetaGuard.project(
                        k=k_norm,
                        b=b,
                        g_cumsum=g,
                        chunk_size=64,
                        cu_seqlens=cu_seqlens,
                        g_is_cumulative=False,
                    )
                    b_native = b + (b_guarded - b).detach()
                b_native = b_native.reshape(-1, b_native.shape[-2], b_native.shape[-1]).contiguous()
                o_native, state_native_out = _cudnn_gdn2(
                    q_native,
                    k_native,
                    v_native,
                    g_native,
                    b_native,
                    w_native,
                    cu_seqlens_native,
                    scale=self.head_k_dim ** -0.5,
                    initial_state=state_native,
                    output_final_state=use_cache,
                    use_qk_l2norm_in_kernel=False,
                )
                o = o_native.reshape(q.shape[0], q.shape[1], o_native.shape[-2], o_native.shape[-1])
                recurrent_state = (
                    state_native_out.transpose(-1, -2).contiguous() if use_cache else None
                )
            else:
                o, recurrent_state = chunk_gdn2(
                    q=q,
                    k=k,
                    v=v,
                    g=g,
                    b=b,
                    w=w,
                    A_log=self.A_log,
                    dt_bias=self.dt_bias,
                    initial_state=recurrent_state,
                    output_final_state=use_cache,
                    use_qk_l2norm_in_kernel=True,
                    use_gate_in_kernel=False,
                    cu_seqlens=cu_seqlens,
                    beta_guard=self.beta_guard,
                    disable_recompute=self.disable_recompute,
                )
        elif mode == "fused_recurrent":
            o, recurrent_state = fused_recurrent_gdn2(
                q=q,
                k=k,
                v=v,
                g=g,
                b=b,
                w=w,
                A_log=self.A_log,
                dt_bias=self.dt_bias,
                initial_state=recurrent_state,
                output_final_state=use_cache,
                use_qk_l2norm_in_kernel=True,
                use_gate_in_kernel=False,
                cu_seqlens=cu_seqlens,
            )
        else:
            raise NotImplementedError(f"Not supported mode `{mode}`.")

        # Persist the recurrent state and short-conv state for the next
        # incremental-decoding step.
        update_layer_cache(
            self,
            past_key_values,
            recurrent_state=recurrent_state,
            conv_state=(conv_state_q, conv_state_k, conv_state_v) if self.use_short_conv else None,
            offset=q_len,
        )

        # SiLU-gated RMS norm on the recurrent output, then project back to
        # the model dimension. Repad if the input batch was unpadded above.
        o = self.o_norm(o, rearrange(output_gate, "... (h d) -> ... h d", d=self.head_v_dim))
        o = rearrange(o, "b t h d -> b t (h d)")
        o = self.o_proj(o)
        if attention_mask is not None:
            o = pad_input(o.squeeze(0), indices, batch_size, q_len)

        return o, None, past_key_values

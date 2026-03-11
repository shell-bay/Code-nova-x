"""
Coding Nova X — Transformer Block
===================================
Each transformer layer = Attention + FFN wrapped in residual connections.

Design decisions:
1. RMSNorm instead of LayerNorm → faster, used in LLaMA/Mistral
2. Pre-normalization (norm before sublayer) → more stable training
3. SwiGLU activation → best FFN quality (used in PaLM, LLaMA)
4. No bias in linear layers → cleaner, modern practice

SwiGLU: output = SiLU(gate) * up  (gated linear unit)
This "gate" controls how much information flows through the FFN.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from model.attention.causal_attention import CausalSelfAttention


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    
    Faster than LayerNorm: no mean subtraction, just RMS scaling.
    Formula: x / RMS(x) * weight
    Used in: LLaMA, Mistral, T5, Gemma
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute RMS in float32 for numerical stability
        x_float = x.float()
        rms = torch.rsqrt(x_float.pow(2).mean(-1, keepdim=True) + self.eps)
        x_normalized = x_float * rms
        return (x_normalized * self.weight.float()).to(x.dtype)


class SwiGLUFFN(nn.Module):
    """
    Feed-Forward Network with SwiGLU activation.
    
    Architecture:
        gate_proj: hidden → intermediate  (gating signal)
        up_proj:   hidden → intermediate  (information)
        down_proj: intermediate → hidden  (projection back)
        
        output = down_proj(SiLU(gate_proj(x)) * up_proj(x))
    
    Why this works: the gate learns WHICH information to let through,
    making the network more selective and expressive.
    """

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

        # Scaled init for deep networks
        std = 0.02
        nn.init.normal_(self.gate_proj.weight, std=std)
        nn.init.normal_(self.up_proj.weight, std=std)
        nn.init.normal_(self.down_proj.weight, std=std / math.sqrt(2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: silu(gate) acts as a smooth gate on the up projection
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


class TransformerBlock(nn.Module):
    """
    Single transformer decoder block.
    
    Flow:
        x → RMSNorm → Attention → + residual
          → RMSNorm → FFN       → + residual
    
    The residual connections (+) are crucial — they let gradients
    flow directly backwards through deep networks (solves vanishing gradient).
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        intermediate_size: int,
        max_seq_len: int = 4096,
        rope_theta: float = 10000.0,
        rms_norm_eps: float = 1e-5,
        attn_dropout: float = 0.0,
        hidden_dropout: float = 0.0,
        use_flash_attn: bool = True,
        layer_idx: int = 0,
    ):
        super().__init__()
        self.layer_idx = layer_idx

        # Pre-attention normalization
        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        
        # Multi-head attention
        self.self_attn = CausalSelfAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            max_seq_len=max_seq_len,
            rope_theta=rope_theta,
            attn_dropout=attn_dropout,
            use_flash_attn=use_flash_attn,
        )

        # Pre-FFN normalization
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        
        # Feed-forward network
        self.mlp = SwiGLUFFN(hidden_size, intermediate_size)

        # Optional dropout on residual
        self.hidden_dropout = nn.Dropout(hidden_dropout) if hidden_dropout > 0 else nn.Identity()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple]]:
        """
        Args:
            hidden_states: [B, seq_len, hidden_size]
            attention_mask: optional bias mask [B, 1, seq_len, seq_len]
            past_key_value: (K, V) from previous generate step
            use_cache: return KV cache for autoregressive generation
        
        Returns:
            hidden_states: [B, seq_len, hidden_size]
            present_kv: tuple of (K, V) or None
        """
        
        # ── Attention sublayer (with pre-norm + residual) ──────────────────
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_out, present_kv = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        hidden_states = self.hidden_dropout(attn_out) + residual

        # ── FFN sublayer (with pre-norm + residual) ────────────────────────
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        ffn_out = self.mlp(hidden_states)
        hidden_states = self.hidden_dropout(ffn_out) + residual

        return hidden_states, present_kv


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    block = TransformerBlock(
        hidden_size=512,
        num_heads=8,
        num_kv_heads=4,
        intermediate_size=2048,
        max_seq_len=2048,
        use_flash_attn=False,
        layer_idx=0,
    ).to(device)

    x = torch.randn(2, 128, 512, device=device)
    out, kv = block(x)
    print(f"[Block] Input:  {x.shape}")
    print(f"[Block] Output: {out.shape}")
    params = sum(p.numel() for p in block.parameters())
    print(f"[Block] Parameters per block: {params:,}")
    print("[Block] ✓ Transformer block working!")

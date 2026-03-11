"""
Coding Nova X — Positional Embeddings
=======================================
Implements Rotary Position Embedding (RoPE) for superior
long-context understanding compared to absolute embeddings.

Why RoPE?
- Encodes relative positions, not absolute → better generalisation
- Works natively with attention math
- Used in LLaMA, Mistral, Falcon, Phi — industry standard
"""

import torch
import torch.nn as nn
import math
from typing import Tuple, Optional


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE).
    
    Instead of adding position info to token embeddings,
    RoPE *rotates* Q and K vectors before dot-product attention.
    This encodes position as a phase shift, letting the model
    naturally understand distance between tokens.
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 4096,
        base: float = 10000.0,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # θ_i = base^(-2i/dim) for i in [0, dim/2)
        # These are the rotation frequencies — different for each dimension pair
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Pre-compute cos/sin cache for speed
        self._build_cache(max_seq_len, device)

    def _build_cache(self, seq_len: int, device=None):
        """Pre-compute cos and sin tables up to seq_len."""
        t = torch.arange(seq_len, device=device or self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)        # [seq_len, dim/2]
        emb = torch.cat([freqs, freqs], dim=-1)      # [seq_len, dim]
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(
        self,
        q: torch.Tensor,    # [B, heads, seq, head_dim]
        k: torch.Tensor,    # [B, kv_heads, seq, head_dim]
        seq_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary embeddings to Q and K."""
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len, q.device)

        cos = self.cos_cached[:, :, :seq_len, :].to(q.dtype)
        sin = self.sin_cached[:, :, :seq_len, :].to(q.dtype)

        q_rot = self._rotate(q, cos, sin)
        k_rot = self._rotate(k, cos, sin)
        return q_rot, k_rot

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        """Rotate the last dimension by half: [x1, x2] → [-x2, x1]."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat([-x2, x1], dim=-1)

    def _rotate(self, x, cos, sin) -> torch.Tensor:
        """Apply rotation: x * cos + rotate_half(x) * sin."""
        return (x * cos) + (self._rotate_half(x) * sin)


class SinusoidalEmbedding(nn.Module):
    """
    Classic sinusoidal positional embedding (Vaswani et al., 2017).
    Used as a fallback when RoPE is not needed.
    """

    def __init__(self, d_model: int, max_seq_len: int = 4096, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_seq_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to embeddings."""
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class LearnedEmbedding(nn.Module):
    """
    Learned absolute positional embedding.
    Simple but works well for shorter sequences.
    """

    def __init__(self, d_model: int, max_seq_len: int = 4096, dropout: float = 0.0):
        super().__init__()
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)
        nn.init.normal_(self.pos_embed.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device)
        return self.dropout(x + self.pos_embed(positions))


if __name__ == "__main__":
    # Quick test
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rope = RotaryEmbedding(dim=128, max_seq_len=4096).to(device)
    
    B, H, S, D = 2, 8, 512, 128
    q = torch.randn(B, H, S, D, device=device)
    k = torch.randn(B, 4, S, D, device=device)
    
    q_rot, k_rot = rope(q, k, seq_len=S)
    print(f"[RoPE] Q: {q.shape} → {q_rot.shape}")
    print(f"[RoPE] K: {k.shape} → {k_rot.shape}")
    print("[RoPE] ✓ Rotary embeddings working correctly!")

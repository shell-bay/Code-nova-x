"""
Coding Nova X — Causal Multi-Head Attention
=============================================
Implements Grouped Query Attention (GQA) + RoPE.

Why GQA?
- Standard MHA: every head has its own K and V → huge KV cache
- GQA: multiple Q heads share K/V heads → smaller cache, same quality
- Used in LLaMA 2/3, Mistral, Gemma — makes inference practical

Architecture:
    Q: num_heads projections  (e.g. 16 heads)
    K: num_kv_heads projections (e.g. 8 heads)  ← shared
    V: num_kv_heads projections (e.g. 8 heads)  ← shared
    
    Each K/V head is shared by (num_heads // num_kv_heads) Q heads.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

# Attempt to use Flash Attention 2 for speed
try:
    from flash_attn import flash_attn_func
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False


class CausalSelfAttention(nn.Module):
    """
    Grouped Query Attention with RoPE and causal masking.
    
    Causal = each token only attends to itself and previous tokens.
    This is how GPT-style models work: they predict next token
    based only on past tokens (no cheating by looking at future).
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_seq_len: int = 4096,
        rope_theta: float = 10000.0,
        attn_dropout: float = 0.0,
        use_flash_attn: bool = True,
    ):
        super().__init__()
        
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        assert num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_heads
        self.kv_groups = num_heads // num_kv_heads   # How many Q heads share each KV head
        self.scale = self.head_dim ** -0.5            # 1/√d_k for stable gradients
        self.use_flash = use_flash_attn and FLASH_ATTN_AVAILABLE
        self.attn_dropout = attn_dropout

        # Linear projections — no bias (like modern LLMs)
        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        # RoPE
        from model.embeddings.rope import RotaryEmbedding
        self.rotary_emb = RotaryEmbedding(
            dim=self.head_dim,
            max_seq_len=max_seq_len,
            base=rope_theta,
        )

        # Causal mask — lower triangular matrix
        # Registered as buffer so it moves with .to(device)
        causal_mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool))
        self.register_buffer("causal_mask", causal_mask.view(1, 1, max_seq_len, max_seq_len))

        # Init weights
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.q_proj.weight, std=0.02)
        nn.init.normal_(self.k_proj.weight, std=0.02)
        nn.init.normal_(self.v_proj.weight, std=0.02)
        nn.init.normal_(self.o_proj.weight, std=0.02 / math.sqrt(2))  # Scaled for depth

    def _expand_kv(self, x: torch.Tensor) -> torch.Tensor:
        """
        Expand KV heads to match Q heads for GQA.
        [B, num_kv_heads, S, D] → [B, num_heads, S, D]
        Each KV head is repeated kv_groups times.
        """
        if self.kv_groups == 1:
            return x
        B, kv_h, S, D = x.shape
        x = x.unsqueeze(2)                                        # [B, kv_h, 1, S, D]
        x = x.expand(B, kv_h, self.kv_groups, S, D)             # [B, kv_h, groups, S, D]
        return x.reshape(B, self.num_heads, S, D)                # [B, num_heads, S, D]

    def forward(
        self,
        hidden_states: torch.Tensor,          # [B, seq_len, hidden_size]
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple] = None,  # For KV cache during inference
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple]]:

        B, S, _ = hidden_states.shape

        # Project to Q, K, V
        q = self.q_proj(hidden_states)   # [B, S, num_heads * head_dim]
        k = self.k_proj(hidden_states)   # [B, S, num_kv_heads * head_dim]
        v = self.v_proj(hidden_states)   # [B, S, num_kv_heads * head_dim]

        # Reshape: [B, S, heads*dim] → [B, heads, S, dim]
        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE to Q and K
        q, k = self.rotary_emb(q, k, seq_len=S)

        # KV Cache: append to past keys/values during inference
        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        present_kv = (k, v) if use_cache else None
        kv_seq_len = k.shape[2]

        # Expand K, V for GQA
        k = self._expand_kv(k)   # [B, num_heads, kv_seq_len, head_dim]
        v = self._expand_kv(v)

        # ── Attention computation ──────────────────────────────────────────
        if self.use_flash and not use_cache:
            # Flash Attention 2 — much faster, uses less memory
            # Needs [B, S, heads, dim] format
            q_fa = q.transpose(1, 2)   # [B, S, heads, dim]
            k_fa = k.transpose(1, 2)
            v_fa = v.transpose(1, 2)
            dropout_p = self.attn_dropout if self.training else 0.0
            attn_out = flash_attn_func(q_fa, k_fa, v_fa, dropout_p=dropout_p, causal=True)
            attn_out = attn_out.reshape(B, S, self.hidden_size)
        else:
            # Standard scaled dot-product attention
            # scores: [B, heads, S, kv_len]
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

            # Apply causal mask — mask out future positions
            causal = self.causal_mask[:, :, :S, :kv_seq_len]
            scores = scores.masked_fill(~causal, float("-inf"))

            # Optional extra attention mask (for padding)
            if attention_mask is not None:
                scores = scores + attention_mask

            # Softmax + dropout
            attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
            if self.training and self.attn_dropout > 0:
                attn_weights = F.dropout(attn_weights, p=self.attn_dropout)

            # Weighted sum of values
            attn_out = torch.matmul(attn_weights, v)   # [B, heads, S, dim]
            attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, self.hidden_size)

        # Output projection
        output = self.o_proj(attn_out)
        return output, present_kv


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    attn = CausalSelfAttention(
        hidden_size=512,
        num_heads=8,
        num_kv_heads=4,
        max_seq_len=2048,
        use_flash_attn=False,
    ).to(device)

    x = torch.randn(2, 128, 512, device=device)
    out, _ = attn(x)
    print(f"[Attention] Input:  {x.shape}")
    print(f"[Attention] Output: {out.shape}")
    params = sum(p.numel() for p in attn.parameters())
    print(f"[Attention] Parameters: {params:,}")
    print("[Attention] ✓ Causal GQA attention working!")

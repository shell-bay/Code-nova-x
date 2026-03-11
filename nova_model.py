"""
Coding Nova X — Complete Transformer Language Model
=====================================================
Full GPT-style decoder-only transformer.

Architecture summary:
  Token Embeddings (vocab → hidden)
  ↓
  N × TransformerBlock (attention + FFN)
  ↓
  Final RMSNorm
  ↓
  LM Head (hidden → vocab logits)

Training objective: next-token prediction (cross-entropy loss)
The model reads a sequence and predicts the next token at every position.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import json
from typing import Optional, List, Tuple, Dict
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


class CodingNovaX(nn.Module):
    """
    Coding Nova X — Decoder-Only Transformer Language Model.
    
    Designed for code generation and understanding.
    Based on GPT architecture with modern improvements:
    - RoPE positional embeddings
    - Grouped Query Attention  
    - SwiGLU feed-forward
    - RMSNorm
    - No bias in linear layers
    """

    def __init__(self, config):
        """
        Args:
            config: ModelConfig dataclass with all hyperparameters
        """
        super().__init__()
        self.config = config

        # ── Token embedding table ─────────────────────────────────────────
        # Maps each token ID to a learned vector of size hidden_size
        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=config.pad_token_id,
        )

        # ── Stack of transformer blocks ───────────────────────────────────
        from model.blocks.transformer_block import TransformerBlock
        self.layers = nn.ModuleList([
            TransformerBlock(
                hidden_size=config.hidden_size,
                num_heads=config.num_attention_heads,
                num_kv_heads=config.num_kv_heads,
                intermediate_size=config.intermediate_size,
                max_seq_len=config.max_position_embeddings,
                rope_theta=config.rope_theta,
                rms_norm_eps=config.rms_norm_eps,
                attn_dropout=config.attention_dropout,
                hidden_dropout=config.hidden_dropout,
                use_flash_attn=config.use_flash_attention,
                layer_idx=i,
            )
            for i in range(config.num_layers)
        ])

        # ── Final layer norm ──────────────────────────────────────────────
        from model.blocks.transformer_block import RMSNorm
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # ── Language model head ───────────────────────────────────────────
        # Projects hidden_size → vocab_size (unnormalized logits)
        if config.tie_word_embeddings:
            # Weight tying: share embed_tokens and lm_head weights
            # Saves memory and often improves quality
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize all weights
        self.apply(self._init_weights)
        
        # Special scaled init for output projections (GPT-2 trick)
        for name, p in self.named_parameters():
            if "o_proj" in name or "down_proj" in name:
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * config.num_layers))

        print(f"[CodingNovaX] Model initialized with {self.count_parameters()/1e6:.1f}M parameters")

    def _init_weights(self, module):
        """Weight initialization following GPT-2 style."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_input_embeddings(self):
        return self.embed_tokens

    def forward(
        self,
        input_ids: torch.Tensor,                        # [B, seq_len]
        attention_mask: Optional[torch.Tensor] = None,  # [B, seq_len]
        labels: Optional[torch.Tensor] = None,          # [B, seq_len] for training
        past_key_values: Optional[List[Tuple]] = None,  # KV cache list
        use_cache: bool = False,
        return_hidden_states: bool = False,
    ) -> Dict:
        """
        Forward pass.
        
        Returns dict with:
            loss: scalar (if labels provided)
            logits: [B, seq_len, vocab_size]
            past_key_values: list of (K, V) tuples (if use_cache=True)
            hidden_states: list of layer outputs (if return_hidden_states=True)
        """
        B, S = input_ids.shape

        # ── Embed tokens ─────────────────────────────────────────────────
        hidden_states = self.embed_tokens(input_ids)   # [B, S, hidden_size]

        # ── Process through transformer layers ───────────────────────────
        all_hidden_states = [] if return_hidden_states else None
        present_key_values = [] if use_cache else None

        for i, layer in enumerate(self.layers):
            if return_hidden_states:
                all_hidden_states.append(hidden_states)

            past_kv = past_key_values[i] if past_key_values is not None else None

            hidden_states, present_kv = layer(
                hidden_states,
                attention_mask=None,   # Causal mask is inside attention module
                past_key_value=past_kv,
                use_cache=use_cache,
            )

            if use_cache:
                present_key_values.append(present_kv)

        # ── Final normalization ───────────────────────────────────────────
        hidden_states = self.norm(hidden_states)

        if return_hidden_states:
            all_hidden_states.append(hidden_states)

        # ── LM head: predict next token logits ───────────────────────────
        if self.lm_head is not None:
            logits = self.lm_head(hidden_states)
        else:
            # Tied weights: transpose of embedding matrix
            logits = F.linear(hidden_states, self.embed_tokens.weight)

        # ── Compute loss if labels provided ───────────────────────────────
        loss = None
        if labels is not None:
            # Shift: predict token at position i+1 from position i
            # logits: [B, S-1, vocab] vs labels: [B, S-1]
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=self.config.pad_token_id,
                reduction="mean",
            )

        return {
            "loss": loss,
            "logits": logits,
            "past_key_values": present_key_values,
            "hidden_states": all_hidden_states,
        }

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        do_sample: bool = True,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Autoregressive text generation.
        
        Sampling strategies:
        - Temperature: controls randomness (lower = more deterministic)
        - Top-K: only sample from K most likely tokens
        - Top-P (nucleus): only sample from tokens covering P% probability
        """
        self.eval()
        eos_id = eos_token_id or self.config.eos_token_id
        pad_id = pad_token_id or self.config.pad_token_id

        generated = input_ids.clone()
        past_key_values = None

        for _ in range(max_new_tokens):
            # Use KV cache for efficiency: only process new tokens
            if past_key_values is not None:
                input_slice = generated[:, -1:]   # Only last token
            else:
                input_slice = generated

            outputs = self(
                input_slice,
                past_key_values=past_key_values,
                use_cache=True,
            )

            logits = outputs["logits"][:, -1, :]       # [B, vocab_size]
            past_key_values = outputs["past_key_values"]

            # Apply temperature scaling
            if temperature != 1.0:
                logits = logits / temperature

            # Apply top-K filtering
            if top_k > 0:
                top_k = min(top_k, logits.size(-1))
                values, _ = torch.topk(logits, top_k)
                min_val = values[:, -1].unsqueeze(-1)
                logits = logits.masked_fill(logits < min_val, float("-inf"))

            # Apply top-P (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                # Remove tokens above cumulative probability threshold
                sorted_remove = cum_probs - F.softmax(sorted_logits, dim=-1) > top_p
                sorted_logits[sorted_remove] = float("-inf")
                logits = torch.zeros_like(logits).scatter_(1, sorted_idx, sorted_logits)

            # Sample or greedy decode
            if do_sample:
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = logits.argmax(dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token], dim=1)

            # Stop if all sequences have generated EOS
            if eos_id is not None and (next_token == eos_id).all():
                break

        return generated

    def save_pretrained(self, save_dir: str):
        """Save model weights and config."""
        os.makedirs(save_dir, exist_ok=True)
        
        # Save config
        with open(os.path.join(save_dir, "config.json"), "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)
        
        # Save weights
        torch.save(self.state_dict(), os.path.join(save_dir, "model.pt"))
        print(f"[CodingNovaX] Saved model to {save_dir}")

    @classmethod
    def from_pretrained(cls, load_dir: str, config=None):
        """Load model from saved directory."""
        if config is None:
            from configs.model_config import ModelConfig
            config = ModelConfig.load(os.path.join(load_dir, "config.json"))
        
        model = cls(config)
        state = torch.load(os.path.join(load_dir, "model.pt"), map_location="cpu")
        model.load_state_dict(state)
        print(f"[CodingNovaX] Loaded model from {load_dir}")
        return model


if __name__ == "__main__":
    from configs.model_config import get_300m_config
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = get_300m_config()
    config.use_flash_attention = False  # Disable for CPU test

    model = CodingNovaX(config).to(device)

    # Test forward pass
    input_ids = torch.randint(0, config.vocab_size, (2, 64), device=device)
    labels = input_ids.clone()

    out = model(input_ids, labels=labels)
    print(f"[Model] Loss: {out['loss'].item():.4f}")
    print(f"[Model] Logits: {out['logits'].shape}")
    print(f"[Model] Parameters: {model.count_parameters()/1e6:.1f}M")
    print("[Model] ✓ CodingNovaX forward pass working!")

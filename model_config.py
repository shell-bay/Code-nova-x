"""
Coding Nova X - Model Configuration
====================================
Central configuration for the entire model system.
All hyperparameters and settings live here.
"""

from dataclasses import dataclass, field
from typing import Optional, List
import json
import os


@dataclass
class ModelConfig:
    """
    Core transformer model configuration.
    Targets ~300M-1B parameters depending on scaling.
    """
    # --- Vocabulary ---
    vocab_size: int = 32000          # SentencePiece vocabulary size
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2

    # --- Architecture ---
    hidden_size: int = 2048          # Embedding / hidden dimension
    num_layers: int = 24             # Transformer decoder blocks
    num_attention_heads: int = 16    # Multi-head attention heads
    num_kv_heads: int = 8            # Grouped Query Attention (GQA) heads
    intermediate_size: int = 8192    # FFN hidden dimension (4x hidden_size)
    max_position_embeddings: int = 4096   # Max context length
    rope_theta: float = 10000.0      # RoPE base frequency

    # --- Normalization ---
    rms_norm_eps: float = 1e-5       # RMSNorm epsilon
    layer_norm_type: str = "rms"     # "rms" or "layer"

    # --- Activation ---
    hidden_act: str = "silu"         # Activation: silu (SwiGLU), gelu, relu
    
    # --- Dropout ---
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    
    # --- Initialization ---
    initializer_range: float = 0.02
    tie_word_embeddings: bool = False

    # --- Training flags ---
    use_flash_attention: bool = True   # Flash Attention 2 if available
    use_gradient_checkpointing: bool = True

    def to_dict(self):
        return self.__dict__

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"[Config] Saved model config to {path}")

    @classmethod
    def load(cls, path: str):
        with open(path, "r") as f:
            data = json.load(f)
        cfg = cls()
        for k, v in data.items():
            setattr(cfg, k, v)
        print(f"[Config] Loaded model config from {path}")
        return cfg

    def estimate_parameters(self) -> int:
        """Roughly estimate model parameter count."""
        # Embeddings
        embed = self.vocab_size * self.hidden_size
        # Per layer: attention (Q,K,V,O) + FFN (gate, up, down) + norms
        attn = (
            self.hidden_size * self.hidden_size +   # Q
            (self.hidden_size // self.num_attention_heads) * self.num_kv_heads * self.hidden_size +  # K
            (self.hidden_size // self.num_attention_heads) * self.num_kv_heads * self.hidden_size +  # V
            self.hidden_size * self.hidden_size      # O
        )
        ffn = (
            self.hidden_size * self.intermediate_size +  # gate
            self.hidden_size * self.intermediate_size +  # up
            self.intermediate_size * self.hidden_size    # down
        )
        norms = self.hidden_size * 4  # 2 norms per layer
        per_layer = attn + ffn + norms
        # LM head
        lm_head = self.vocab_size * self.hidden_size if not self.tie_word_embeddings else 0
        total = embed + (per_layer * self.num_layers) + lm_head
        return total


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    # --- Batching ---
    batch_size: int = 4               # Per-GPU batch size
    gradient_accumulation_steps: int = 16  # Effective batch = 4*16=64
    max_seq_len: int = 4096

    # --- Optimizer ---
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    max_grad_norm: float = 1.0

    # --- Schedule ---
    lr_scheduler: str = "cosine"      # cosine, linear, constant
    warmup_steps: int = 2000
    max_steps: int = 500000           # Total training steps

    # --- Precision ---
    dtype: str = "bfloat16"           # bfloat16 or float16 or float32
    use_amp: bool = True

    # --- Checkpointing ---
    save_every_steps: int = 1000
    eval_every_steps: int = 500
    checkpoint_dir: str = "checkpoints"
    keep_last_n_checkpoints: int = 3

    # --- Logging ---
    log_every_steps: int = 10
    log_dir: str = "logs"

    # --- Data ---
    dataset_path: str = "data/tokenized"
    num_workers: int = 4

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.__dict__, f, indent=2)


@dataclass
class TokenizerConfig:
    """Tokenizer training configuration."""
    vocab_size: int = 32000
    model_type: str = "bpe"           # bpe or unigram
    character_coverage: float = 0.9999
    input_files: List[str] = field(default_factory=list)
    model_prefix: str = "tokenizer/nova_tokenizer"
    
    # Special tokens
    pad_piece: str = "<pad>"
    bos_piece: str = "<s>"
    eos_piece: str = "</s>"
    unk_piece: str = "<unk>"
    
    # Code special tokens
    extra_special_tokens: List[str] = field(default_factory=lambda: [
        "<|code|>", "<|endcode|>",
        "<|python|>", "<|cpp|>", "<|javascript|>",
        "<|java|>", "<|rust|>", "<|bash|>",
        "<|human|>", "<|assistant|>",
        "<|system|>", "<|endoftext|>",
        "<|debug|>", "<|fix|>", "<|explain|>",
    ])


# ── Default configs ──────────────────────────────────────────────────────────

def get_300m_config() -> ModelConfig:
    """~300M parameter configuration."""
    return ModelConfig(
        vocab_size=32000,
        hidden_size=1024,
        num_layers=24,
        num_attention_heads=16,
        num_kv_heads=8,
        intermediate_size=4096,
        max_position_embeddings=4096,
    )


def get_700m_config() -> ModelConfig:
    """~700M parameter configuration."""
    return ModelConfig(
        vocab_size=32000,
        hidden_size=2048,
        num_layers=24,
        num_attention_heads=16,
        num_kv_heads=8,
        intermediate_size=8192,
        max_position_embeddings=4096,
    )


def get_1b_config() -> ModelConfig:
    """~1B parameter configuration."""
    return ModelConfig(
        vocab_size=32000,
        hidden_size=2048,
        num_layers=32,
        num_attention_heads=32,
        num_kv_heads=8,
        intermediate_size=8192,
        max_position_embeddings=4096,
    )


if __name__ == "__main__":
    cfg = get_700m_config()
    params = cfg.estimate_parameters()
    print(f"[Coding Nova X] Estimated parameters: {params/1e6:.1f}M")
    cfg.save("configs/model_700m.json")

    train_cfg = TrainingConfig()
    train_cfg.save("configs/training.json")
    print("[Coding Nova X] Configuration files saved!")

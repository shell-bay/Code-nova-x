"""
Coding Nova X — Instruction Fine-Tuning
=========================================
Fine-tunes the pretrained model to follow instructions,
generate code from prompts, debug code, and explain concepts.

This is what makes the model USEFUL — raw pretraining gives
language understanding, but fine-tuning teaches HOW to respond.

Training format (Alpaca-style):
    <|system|>
    You are Coding Nova X...
    <|human|>
    Write a function to sort a list
    <|assistant|>
    def sort_list(lst):
        return sorted(lst)
    </s>

Loss only computed on assistant tokens (not system/human).
"""

import os
import sys
import json
import math
from typing import List, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


SYSTEM_PROMPT = """You are Coding Nova X, an expert AI assistant specialized in programming. 

You can:
- Write clean, efficient, well-documented code in any language
- Debug and fix broken code with clear explanations  
- Explain programming concepts in simple terms
- Suggest optimal algorithms and data structures
- Review code and suggest improvements

Always provide working code with helpful comments."""


def load_finetune_dataset(path: Optional[str] = None) -> List[Dict]:
    """
    Load fine-tuning dataset.
    Supports: JSON array, JSONL, or built-in data.
    """
    from dataset.data_pipeline import get_builtin_instruction_data
    
    data = get_builtin_instruction_data()
    
    if path and os.path.exists(path):
        with open(path) as f:
            if path.endswith('.jsonl'):
                extra = [json.loads(line) for line in f if line.strip()]
            else:
                extra = json.load(f)
        data.extend(extra)
        print(f"[Finetune] Loaded {len(extra)} extra samples from {path}")
    
    print(f"[Finetune] Total fine-tuning samples: {len(data)}")
    return data


def finetune(
    base_model_path: Optional[str] = None,
    data_path: Optional[str] = None,
    output_dir: str = "checkpoints/finetuned",
    epochs: int = 3,
    batch_size: int = 2,
    learning_rate: float = 2e-5,   # Much lower than pretraining!
    max_seq_len: int = 2048,
):
    """
    Fine-tune Coding Nova X on instruction data.
    
    Why lower LR than pretraining?
    The model already knows language — we just want to adjust
    its behavior slightly, not overwrite what it learned.
    Too high LR → "catastrophic forgetting" of pretraining knowledge.
    """
    from configs.model_config import get_300m_config, TrainingConfig
    from model.nova_model import CodingNovaX
    from tokenizer.nova_tokenizer import NovaTokenizer
    from dataset.data_pipeline import InstructionDataset, collate_fn
    from training.trainer import NovaTrainer, setup_logger

    logger = setup_logger("logs", "nova_finetune")
    logger.info("=" * 60)
    logger.info("  Coding Nova X — Instruction Fine-Tuning")
    logger.info("=" * 60)

    # ── Load tokenizer ────────────────────────────────────────────────────────
    tok_path = "tokenizer/nova_tokenizer.model"
    if not os.path.exists(tok_path):
        raise FileNotFoundError(f"Tokenizer not found at {tok_path}. Run pretraining first!")
    tokenizer = NovaTokenizer(tok_path)

    # ── Load model ────────────────────────────────────────────────────────────
    if base_model_path and os.path.exists(os.path.join(base_model_path, "model.pt")):
        logger.info(f"Loading base model from {base_model_path}")
        model = CodingNovaX.from_pretrained(base_model_path)
    else:
        logger.info("No base model found, initializing fresh model...")
        cfg = get_300m_config()
        cfg.vocab_size = tokenizer.vocab_size
        cfg.use_flash_attention = False
        model = CodingNovaX(cfg)

    # ── Load data ─────────────────────────────────────────────────────────────
    pairs = load_finetune_dataset(data_path)
    
    # Split train/eval
    split = int(len(pairs) * 0.9)
    train_pairs = pairs[:split]
    eval_pairs = pairs[split:] if split < len(pairs) else pairs[:1]

    train_dataset = InstructionDataset(
        train_pairs, tokenizer, max_seq_len, system_prompt=SYSTEM_PROMPT
    )
    eval_dataset = InstructionDataset(
        eval_pairs, tokenizer, max_seq_len, system_prompt=SYSTEM_PROMPT
    )

    pad_id = tokenizer.pad_id
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=lambda b: collate_fn(b, pad_id, max_seq_len),
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda b: collate_fn(b, pad_id, max_seq_len),
    )

    # ── Training config ───────────────────────────────────────────────────────
    total_steps = epochs * len(train_loader)
    
    train_cfg = TrainingConfig(
        batch_size=batch_size,
        gradient_accumulation_steps=2,
        max_steps=total_steps,
        learning_rate=learning_rate,
        warmup_steps=max(10, total_steps // 20),
        weight_decay=0.01,
        checkpoint_dir=output_dir,
        max_seq_len=max_seq_len,
        log_every_steps=5,
        eval_every_steps=50,
        save_every_steps=100,
    )

    # ── Fine-tune ─────────────────────────────────────────────────────────────
    trainer = NovaTrainer(model, train_cfg, logger)
    trainer.train(train_loader, eval_loader)

    # ── Save final model ──────────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    logger.info(f"Fine-tuned model saved to {output_dir}")
    return model


if __name__ == "__main__":
    print("Run: from training.finetune import finetune; finetune()")

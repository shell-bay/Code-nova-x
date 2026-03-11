"""
Coding Nova X — Training Engine
==================================
Production-grade training loop with:
- Mixed precision (BF16/FP16)
- Gradient accumulation (train on large batches with limited VRAM)
- Gradient clipping (prevent exploding gradients)
- Cosine LR schedule with warmup
- Checkpoint saving/loading
- TensorBoard + console logging
- Automatic recovery from crashes
"""

import os
import sys
import time
import math
import json
import logging
from typing import Optional, Dict, List
from dataclasses import asdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


# ── Logging setup ─────────────────────────────────────────────────────────────

def setup_logger(log_dir: str = "logs", name: str = "nova_trainer") -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    fmt = logging.Formatter('[%(asctime)s] %(levelname)s — %(message)s', datefmt='%H:%M:%S')
    
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    
    fh = logging.FileHandler(os.path.join(log_dir, f"{name}.log"))
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    
    return logger


# ── Learning Rate Scheduler ───────────────────────────────────────────────────

class CosineScheduler:
    """
    Cosine learning rate schedule with linear warmup.
    
    Phase 1: Linear warmup from 0 to max_lr over warmup_steps
    Phase 2: Cosine decay from max_lr to min_lr over remaining steps
    
    Why cosine? Smooth decay helps model converge to better minima.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        max_lr: float,
        warmup_steps: int,
        max_steps: int,
        min_lr_ratio: float = 0.1,
    ):
        self.optimizer = optimizer
        self.max_lr = max_lr
        self.min_lr = max_lr * min_lr_ratio
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.current_step = 0

    def step(self):
        """Update optimizer LR for current step."""
        self.current_step += 1
        lr = self._compute_lr(self.current_step)
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
        return lr

    def _compute_lr(self, step: int) -> float:
        if step < self.warmup_steps:
            # Linear warmup
            return self.max_lr * step / max(1, self.warmup_steps)
        
        # Cosine decay
        progress = (step - self.warmup_steps) / max(1, self.max_steps - self.warmup_steps)
        progress = min(progress, 1.0)
        cosine_val = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr + (self.max_lr - self.min_lr) * cosine_val


# ── Main Trainer ──────────────────────────────────────────────────────────────

class NovaTrainer:
    """
    Complete training engine for Coding Nova X.
    
    Handles both pretraining (next-token prediction) and
    instruction fine-tuning (instruction-response format).
    """

    def __init__(
        self,
        model: nn.Module,
        train_config,
        logger: Optional[logging.Logger] = None,
    ):
        self.model = model
        self.config = train_config
        self.logger = logger or setup_logger(train_config.log_dir)
        
        self.device = self._get_device()
        self.model = self.model.to(self.device)
        
        # Optimizer: AdamW with decoupled weight decay
        self.optimizer = self._build_optimizer()
        
        # Mixed precision scaler (for FP16)
        self.scaler = GradScaler(enabled=(train_config.dtype == "float16"))
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float('inf')
        
        # Training history for visualization
        self.loss_history = []
        self.lr_history = []

        os.makedirs(train_config.checkpoint_dir, exist_ok=True)
        os.makedirs(train_config.log_dir, exist_ok=True)
        
        self.logger.info(f"[Trainer] Device: {self.device}")
        self.logger.info(f"[Trainer] Model params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
        self.logger.info(f"[Trainer] Batch size: {train_config.batch_size} × accumulation {train_config.gradient_accumulation_steps} = {train_config.batch_size * train_config.gradient_accumulation_steps} effective")

    def _get_device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _build_optimizer(self) -> AdamW:
        """
        AdamW optimizer with weight decay.
        
        Important: Don't apply weight decay to:
        - Bias terms
        - LayerNorm/RMSNorm weights
        These act as regularizers themselves.
        """
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "norm" in name.lower() or "bias" in name.lower() or "embed" in name.lower():
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        
        param_groups = [
            {"params": decay_params, "weight_decay": self.config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        
        return AdamW(
            param_groups,
            lr=self.config.learning_rate,
            betas=(self.config.beta1, self.config.beta2),
            eps=1e-8,
        )

    def train(
        self,
        train_loader: DataLoader,
        eval_loader: Optional[DataLoader] = None,
        resume_from: Optional[str] = None,
    ):
        """
        Main training loop.
        
        Args:
            train_loader: DataLoader for training data
            eval_loader: Optional DataLoader for evaluation
            resume_from: Path to checkpoint to resume from
        """
        if resume_from:
            self._load_checkpoint(resume_from)
        
        # Learning rate scheduler
        scheduler = CosineScheduler(
            optimizer=self.optimizer,
            max_lr=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            max_steps=self.config.max_steps,
        )
        
        # Fast-forward scheduler to current step
        for _ in range(self.global_step):
            scheduler.step()
        
        self.logger.info("=" * 60)
        self.logger.info("  Starting Coding Nova X Training")
        self.logger.info("=" * 60)
        
        self.model.train()
        accumulated_loss = 0.0
        accumulation_counter = 0
        t_start = time.time()
        
        while self.global_step < self.config.max_steps:
            for batch in train_loader:
                if self.global_step >= self.config.max_steps:
                    break
                
                # Move batch to device
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                # ── Forward pass ──────────────────────────────────────────
                dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16}
                amp_dtype = dtype_map.get(self.config.dtype)
                
                with autocast(device_type=str(self.device).split(':')[0], 
                             dtype=amp_dtype, enabled=self.config.use_amp and amp_dtype is not None):
                    outputs = self.model(input_ids=input_ids, labels=labels)
                    loss = outputs["loss"]
                    
                    # Scale loss by accumulation steps
                    # (equivalent to averaging over the full effective batch)
                    scaled_loss = loss / self.config.gradient_accumulation_steps
                
                # ── Backward pass ─────────────────────────────────────────
                if self.scaler.is_enabled():
                    self.scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()
                
                accumulated_loss += loss.item()
                accumulation_counter += 1
                
                # ── Optimizer step (after accumulating gradients) ─────────
                if accumulation_counter == self.config.gradient_accumulation_steps:
                    # Unscale gradients for clipping
                    if self.scaler.is_enabled():
                        self.scaler.unscale_(self.optimizer)
                    
                    # Clip gradients — prevents exploding gradients in deep nets
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.max_grad_norm
                    )
                    
                    if self.scaler.is_enabled():
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    
                    lr = scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    
                    avg_loss = accumulated_loss / accumulation_counter
                    self.loss_history.append(avg_loss)
                    self.lr_history.append(lr)
                    
                    self.global_step += 1
                    accumulated_loss = 0.0
                    accumulation_counter = 0
                    
                    # ── Logging ───────────────────────────────────────────
                    if self.global_step % self.config.log_every_steps == 0:
                        elapsed = time.time() - t_start
                        tokens_seen = self.global_step * self.config.batch_size * self.config.gradient_accumulation_steps * self.config.max_seq_len
                        ppl = math.exp(min(avg_loss, 20))
                        
                        self.logger.info(
                            f"Step {self.global_step:6d}/{self.config.max_steps} | "
                            f"Loss: {avg_loss:.4f} | PPL: {ppl:.1f} | "
                            f"LR: {lr:.2e} | GradNorm: {grad_norm:.3f} | "
                            f"Tokens: {tokens_seen/1e6:.1f}M | "
                            f"Time: {elapsed/60:.1f}min"
                        )
                    
                    # ── Evaluation ────────────────────────────────────────
                    if eval_loader and self.global_step % self.config.eval_every_steps == 0:
                        eval_loss = self.evaluate(eval_loader)
                        self.logger.info(f"[EVAL] Step {self.global_step} | Loss: {eval_loss:.4f} | PPL: {math.exp(min(eval_loss,20)):.2f}")
                        self.model.train()
                        
                        if eval_loss < self.best_loss:
                            self.best_loss = eval_loss
                            self._save_checkpoint("best_model")
                            self.logger.info(f"[EVAL] New best model! Loss: {eval_loss:.4f}")
                    
                    # ── Checkpoint ────────────────────────────────────────
                    if self.global_step % self.config.save_every_steps == 0:
                        self._save_checkpoint(f"step_{self.global_step:07d}")
                        self._cleanup_old_checkpoints()
        
        self.logger.info("=" * 60)
        self.logger.info(f"  Training complete! Steps: {self.global_step}")
        self.logger.info(f"  Best loss: {self.best_loss:.4f}")
        self.logger.info("=" * 60)
        self._save_checkpoint("final_model")
        self._save_training_history()

    @torch.no_grad()
    def evaluate(self, eval_loader: DataLoader, max_batches: int = 50) -> float:
        """Run evaluation and return average loss."""
        self.model.eval()
        total_loss = 0.0
        count = 0
        
        for i, batch in enumerate(eval_loader):
            if i >= max_batches:
                break
            
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)
            
            outputs = self.model(input_ids=input_ids, labels=labels)
            total_loss += outputs["loss"].item()
            count += 1
        
        return total_loss / max(count, 1)

    def _save_checkpoint(self, name: str):
        """Save model, optimizer, and training state."""
        path = os.path.join(self.config.checkpoint_dir, name)
        os.makedirs(path, exist_ok=True)
        
        torch.save({
            "global_step": self.global_step,
            "epoch": self.epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scaler_state": self.scaler.state_dict(),
            "best_loss": self.best_loss,
            "loss_history": self.loss_history[-1000:],  # Keep last 1000
            "lr_history": self.lr_history[-1000:],
        }, os.path.join(path, "checkpoint.pt"))
        
        self.logger.info(f"[Checkpoint] Saved: {path}")

    def _load_checkpoint(self, path: str):
        """Resume training from checkpoint."""
        ckpt_file = os.path.join(path, "checkpoint.pt")
        if not os.path.exists(ckpt_file):
            self.logger.warning(f"[Checkpoint] Not found: {ckpt_file}")
            return
        
        state = torch.load(ckpt_file, map_location=self.device)
        self.model.load_state_dict(state["model_state"])
        self.optimizer.load_state_dict(state["optimizer_state"])
        self.scaler.load_state_dict(state.get("scaler_state", {}))
        self.global_step = state["global_step"]
        self.epoch = state.get("epoch", 0)
        self.best_loss = state.get("best_loss", float('inf'))
        self.loss_history = state.get("loss_history", [])
        self.lr_history = state.get("lr_history", [])
        self.logger.info(f"[Checkpoint] Resumed from step {self.global_step}")

    def _cleanup_old_checkpoints(self):
        """Keep only the last N checkpoints."""
        ckpt_dir = self.config.checkpoint_dir
        step_dirs = sorted([
            d for d in os.listdir(ckpt_dir)
            if d.startswith("step_") and os.path.isdir(os.path.join(ckpt_dir, d))
        ])
        
        while len(step_dirs) > self.config.keep_last_n_checkpoints:
            old_dir = os.path.join(ckpt_dir, step_dirs.pop(0))
            import shutil
            shutil.rmtree(old_dir, ignore_errors=True)

    def _save_training_history(self):
        """Save loss/lr history for visualization."""
        history = {
            "loss": self.loss_history,
            "lr": self.lr_history,
            "final_step": self.global_step,
            "best_loss": self.best_loss,
        }
        with open(os.path.join(self.config.log_dir, "training_history.json"), "w") as f:
            json.dump(history, f)
        self.logger.info("[Trainer] Training history saved.")


# ── Quick-start training function ─────────────────────────────────────────────

def pretrain(
    data_dir: str = "data/processed",
    checkpoint_dir: str = "checkpoints",
    model_size: str = "300m",
    max_steps: int = 1000,
    batch_size: int = 2,
):
    """
    Convenience function to start pretraining.
    Good for testing and small-scale training.
    """
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from configs.model_config import get_300m_config, get_700m_config, TrainingConfig
    from model.nova_model import CodingNovaX
    from tokenizer.nova_tokenizer import NovaTokenizer, train_tokenizer
    from dataset.data_pipeline import build_pretraining_dataloader

    logger = setup_logger()
    logger.info("Starting Coding Nova X Pretraining")

    # 1. Load or train tokenizer
    tok_path = "tokenizer/nova_tokenizer.model"
    if os.path.exists(tok_path):
        tokenizer = NovaTokenizer(tok_path)
    else:
        logger.info("Training tokenizer first...")
        tokenizer = train_tokenizer(vocab_size=32000)

    # 2. Build model
    cfg = get_300m_config() if model_size == "300m" else get_700m_config()
    cfg.vocab_size = tokenizer.vocab_size
    cfg.use_flash_attention = False  # Disable unless flash_attn installed
    model = CodingNovaX(cfg)

    # 3. Training config
    train_cfg = TrainingConfig(
        batch_size=batch_size,
        gradient_accumulation_steps=4,
        max_steps=max_steps,
        learning_rate=3e-4,
        warmup_steps=min(100, max_steps // 10),
        checkpoint_dir=checkpoint_dir,
        max_seq_len=512,  # Shorter for testing
    )

    # 4. Build dataloader
    train_loader = build_pretraining_dataloader(
        data_dir=data_dir,
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_seq_len=512,
        num_workers=0,
    )

    # 5. Train!
    trainer = NovaTrainer(model, train_cfg, logger)
    trainer.train(train_loader)
    return model, tokenizer


if __name__ == "__main__":
    print("Coding Nova X — Trainer ready!")
    print("To start training, run: python training/trainer.py")
    print("Or import pretrain() from this module.")

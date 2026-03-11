"""
Coding Nova X — Model Optimization & Quantization
====================================================
Makes the model smaller and faster for deployment.

Techniques:
1. Dynamic INT8 quantization — 4x smaller, 2-3x faster on CPU
2. Static INT8 quantization — requires calibration data
3. GGUF conversion — for llama.cpp and mobile inference
4. Weight pruning — remove near-zero weights

Why quantize?
- Original model: 700M params × 4 bytes = ~2.8 GB (FP32)
- INT8 quantized: ~700 MB (4x smaller!)
- INT4 quantized: ~350 MB (8x smaller!)
- Runs on phones, laptops, Raspberry Pi
"""

import os
import sys
import json
import struct
import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


class DynamicQuantizer:
    """
    Dynamic INT8 quantization using PyTorch's built-in tools.
    
    "Dynamic" means weights are quantized at load time,
    activations are quantized on-the-fly during inference.
    No calibration data needed — works immediately!
    """

    def __init__(self):
        self.original_size = 0
        self.quantized_size = 0

    def quantize(self, model: nn.Module) -> nn.Module:
        """Apply dynamic INT8 quantization to the model."""
        # Measure original size
        self.original_size = self._model_size_mb(model)
        
        # Apply quantization to linear layers
        quantized = torch.quantization.quantize_dynamic(
            model,
            {nn.Linear},          # Which layer types to quantize
            dtype=torch.qint8,    # 8-bit integer weights
        )
        
        self.quantized_size = self._model_size_mb(quantized)
        ratio = self.original_size / max(self.quantized_size, 0.1)
        
        print(f"[Quantize] Original: {self.original_size:.1f} MB")
        print(f"[Quantize] Quantized: {self.quantized_size:.1f} MB")
        print(f"[Quantize] Compression: {ratio:.1f}x")
        
        return quantized

    def _model_size_mb(self, model: nn.Module) -> float:
        """Estimate model size in MB."""
        total_bytes = sum(
            p.numel() * p.element_size()
            for p in model.parameters()
        )
        return total_bytes / (1024 * 1024)

    def save_quantized(self, model: nn.Module, path: str):
        """Save quantized model."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        torch.save(model.state_dict(), path)
        size = os.path.getsize(path) / (1024*1024)
        print(f"[Quantize] Saved quantized model: {path} ({size:.1f} MB)")


class GGUFConverter:
    """
    Convert model to GGUF format for llama.cpp.
    
    GGUF (GGML Unified Format) is used by:
    - llama.cpp — C++ inference engine
    - Ollama — local model server
    - LM Studio — desktop app
    - Mobile apps
    
    This lets Coding Nova X run on phones and laptops
    without Python or PyTorch installed!
    """

    # GGUF magic number and version
    GGUF_MAGIC = 0x46554747  # "GGUF" in little-endian
    GGUF_VERSION = 3

    def __init__(self, output_dir: str = "deployment/gguf"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def convert(
        self,
        model,
        tokenizer,
        model_config,
        output_name: str = "coding_nova_x.gguf",
        quant_type: str = "Q4_K_M",   # Popular quantization level
    ) -> str:
        """
        Convert PyTorch model to GGUF format.
        
        Quantization levels:
        - Q8_0: 8-bit, high quality, ~1x smaller than FP16
        - Q4_K_M: 4-bit, good balance (recommended for most users)
        - Q4_0: 4-bit, smallest, some quality loss
        - Q2_K: 2-bit, very small, noticeable quality loss
        """
        output_path = os.path.join(self.output_dir, output_name)
        
        print(f"[GGUF] Converting model to GGUF format ({quant_type})...")
        print(f"[GGUF] Output: {output_path}")
        
        # In production, this would use llama.cpp's convert.py script
        # Here we create a proper GGUF-compatible metadata file
        
        metadata = {
            "gguf_version": self.GGUF_VERSION,
            "format": "GGUF",
            "model_name": "coding_nova_x",
            "architecture": "llama",   # Compatible architecture
            "quantization": quant_type,
            "model_config": {
                "vocab_size": model_config.vocab_size,
                "hidden_size": model_config.hidden_size,
                "num_layers": model_config.num_layers,
                "num_heads": model_config.num_attention_heads,
                "num_kv_heads": model_config.num_kv_heads,
                "intermediate_size": model_config.intermediate_size,
                "max_seq_len": model_config.max_position_embeddings,
                "rope_theta": model_config.rope_theta,
            },
            "special_tokens": {
                "bos_id": model_config.bos_token_id,
                "eos_id": model_config.eos_token_id,
                "pad_id": model_config.pad_token_id,
            },
            "instructions": [
                f"Full GGUF conversion requires llama.cpp tools.",
                f"Install llama.cpp: git clone https://github.com/ggerganov/llama.cpp",
                f"Then run: python convert.py --model coding_nova_x --outtype {quant_type}",
                f"This file contains model metadata for reference.",
            ]
        }
        
        # Save metadata
        meta_path = output_path.replace('.gguf', '_metadata.json')
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save quantized weights in a simplified format
        weight_path = output_path.replace('.gguf', '_weights.pt')
        
        # Apply INT4-like quantization (simplified)
        quantized_state = {}
        for name, param in model.state_dict().items():
            if 'weight' in name and param.dim() >= 2:
                # Simulate Q4 quantization
                q_weight = self._quantize_to_int4(param.float())
                quantized_state[name] = q_weight
            else:
                quantized_state[name] = param
        
        torch.save(quantized_state, weight_path)
        
        weight_size = os.path.getsize(weight_path) / (1024*1024)
        print(f"[GGUF] Quantized weights saved: {weight_size:.1f} MB")
        print(f"[GGUF] Metadata saved: {meta_path}")
        print(f"\n[GGUF] For full GGUF conversion, follow instructions in {meta_path}")
        
        return meta_path

    def _quantize_to_int4(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Simplified INT4 quantization.
        Real Q4_K_M uses blocked quantization with scales.
        """
        # Scale to [-8, 7] range (4-bit signed)
        max_val = tensor.abs().max() + 1e-8
        scale = max_val / 7.0
        quantized = torch.clamp(torch.round(tensor / scale), -8, 7)
        # Pack two 4-bit values into one int8
        # (simplified — real packing is more complex)
        return quantized.to(torch.int8)


class WeightPruner:
    """
    Prune near-zero weights to reduce model size and computation.
    
    Pruning removes connections with very small weights
    that contribute little to the model's output.
    
    Warning: Too much pruning hurts quality!
    Start with sparsity=0.1 (10%) and measure quality loss.
    """

    def __init__(self, sparsity: float = 0.1):
        """
        Args:
            sparsity: Fraction of weights to zero out (0.1 = 10%)
        """
        self.sparsity = sparsity
        assert 0 <= sparsity < 1.0, "Sparsity must be in [0, 1)"

    def prune(self, model: nn.Module) -> Dict:
        """Apply magnitude-based pruning."""
        stats = {"total_params": 0, "pruned_params": 0, "layers": {}}
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                with torch.no_grad():
                    weight = module.weight
                    
                    # Find threshold: prune the smallest `sparsity` fraction
                    threshold = torch.quantile(weight.abs().flatten(), self.sparsity)
                    mask = weight.abs() >= threshold
                    
                    pruned = (~mask).sum().item()
                    total = weight.numel()
                    
                    # Zero out pruned weights
                    module.weight.data *= mask.float()
                    
                    stats["total_params"] += total
                    stats["pruned_params"] += pruned
                    stats["layers"][name] = {
                        "total": total,
                        "pruned": pruned,
                        "sparsity": pruned / total,
                    }
        
        actual_sparsity = stats["pruned_params"] / max(stats["total_params"], 1)
        print(f"[Pruner] Actual sparsity: {actual_sparsity*100:.1f}%")
        print(f"[Pruner] Pruned {stats['pruned_params']:,}/{stats['total_params']:,} weights")
        
        return stats


def optimize_model(
    model,
    model_config,
    tokenizer=None,
    output_dir: str = "deployment",
    quantize: bool = True,
    prune: bool = False,
    convert_gguf: bool = True,
    prune_sparsity: float = 0.05,
) -> Dict:
    """
    Run full optimization pipeline.
    
    Args:
        model: Trained CodingNovaX model
        model_config: Model configuration
        tokenizer: Optional tokenizer for GGUF conversion
        output_dir: Where to save optimized models
        quantize: Apply INT8 quantization
        prune: Apply weight pruning
        convert_gguf: Generate GGUF metadata
        prune_sparsity: Fraction to prune (if pruning)
    
    Returns:
        Dict with paths and metrics for all optimized versions
    """
    os.makedirs(output_dir, exist_ok=True)
    results = {}
    
    print("\n" + "=" * 60)
    print("  Coding Nova X — Model Optimization")
    print("=" * 60)
    
    # 1. Pruning (before quantization)
    if prune:
        print(f"\n[Step 1] Weight Pruning (sparsity={prune_sparsity})")
        pruner = WeightPruner(sparsity=prune_sparsity)
        prune_stats = pruner.prune(model)
        results["pruning"] = prune_stats
        
        pruned_path = os.path.join(output_dir, "pruned_model.pt")
        torch.save(model.state_dict(), pruned_path)
        print(f"[Pruning] Saved to {pruned_path}")
    
    # 2. Dynamic INT8 Quantization
    if quantize:
        print("\n[Step 2] Dynamic INT8 Quantization")
        quantizer = DynamicQuantizer()
        
        # Move to CPU for quantization
        model_cpu = model.cpu()
        quantized_model = quantizer.quantize(model_cpu)
        
        quant_path = os.path.join(output_dir, "quantized_int8.pt")
        quantizer.save_quantized(quantized_model, quant_path)
        results["quantization"] = {
            "original_mb": quantizer.original_size,
            "quantized_mb": quantizer.quantized_size,
            "compression_ratio": quantizer.original_size / max(quantizer.quantized_size, 0.1),
            "path": quant_path,
        }
    
    # 3. GGUF conversion
    if convert_gguf:
        print("\n[Step 3] GGUF Conversion Metadata")
        converter = GGUFConverter(os.path.join(output_dir, "gguf"))
        gguf_path = converter.convert(model, tokenizer, model_config)
        results["gguf"] = {"metadata_path": gguf_path}
    
    # Save optimization report
    report_path = os.path.join(output_dir, "optimization_report.json")
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n[Optimize] Report saved: {report_path}")
    
    return results


if __name__ == "__main__":
    print("Coding Nova X Optimizer ready.")
    print("Import optimize_model() and pass your trained model.")

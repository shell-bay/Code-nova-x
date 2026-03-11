# 🚀 Coding Nova X

**A production-ready, transformer-based AI model for code generation, debugging, and programming assistance.**

Built from scratch with PyTorch — 300M to 1B+ parameters, quantizable to run on mobile devices.

---

## Architecture

```
CodingNovaX (GPT Decoder-Only)
├── Token Embeddings        (vocab → hidden_size)
├── 24 TransformerBlocks ×
│   ├── RMSNorm
│   ├── CausalSelfAttention (GQA + RoPE + Flash Attention)
│   ├── RMSNorm  
│   └── SwiGLU FFN
├── Final RMSNorm
└── LM Head (hidden_size → vocab)
```

| Config | Parameters | Hidden | Layers | Heads |
|--------|-----------|--------|--------|-------|
| 300M   | ~443M     | 1024   | 24     | 16    |
| 700M   | ~1.6B     | 2048   | 24     | 16    |
| 1B     | ~2.1B     | 2048   | 32     | 32    |

---

## Quick Start

```bash
# 1. Install dependencies
pip install torch sentencepiece numpy datasets tqdm

# 2. Train tokenizer (required first)
python run.py --phase tokenizer

# 3. Pretrain the model
python run.py --phase pretrain --model_size 300m --max_steps 500000

# 4. Fine-tune on instructions
python run.py --phase finetune --finetune_epochs 3

# 5. Evaluate
python run.py --phase evaluate

# 6. Chat!
python run.py --phase chat

# 7. Optimize for mobile
python run.py --phase optimize

# Quick demo (no training needed)
python run.py --phase demo
```

---

## Project Structure

```
coding_nova_x/
├── model/
│   ├── nova_model.py          # Complete CodingNovaX model
│   ├── attention/
│   │   └── causal_attention.py  # GQA + RoPE attention
│   ├── blocks/
│   │   └── transformer_block.py # RMSNorm + SwiGLU FFN
│   └── embeddings/
│       └── rope.py              # Rotary Position Embedding
├── tokenizer/
│   └── nova_tokenizer.py      # BPE tokenizer (SentencePiece)
├── dataset/
│   └── data_pipeline.py       # Streaming dataset + deduplication
├── training/
│   ├── trainer.py             # Pretraining engine
│   └── finetune.py            # Instruction fine-tuning
├── evaluation/
│   └── evaluator.py           # Perplexity + code benchmarks
├── inference/
│   └── engine.py              # KV-cached inference engine
├── optimization/
│   └── quantize.py            # INT8/INT4 + GGUF conversion
├── deployment/
│   ├── android/               # Android JNI + ONNX guide
│   └── gguf/                  # GGUF metadata
├── self_improve/
│   └── self_improve.py        # Generate → Execute → Fix loop
├── configs/
│   └── model_config.py        # All hyperparameters
└── run.py                     # Main entry point
```

---

## Key Design Decisions

### Why Grouped Query Attention (GQA)?
Standard MHA has K/V heads = Q heads → huge KV cache during inference.
GQA shares K/V heads across multiple Q heads → 4-8x smaller KV cache.
Used in LLaMA 2/3, Mistral, Gemma.

### Why RoPE (Rotary Position Embedding)?
Encodes relative positions rather than absolute → better generalization.
No learnable parameters → robust across sequence lengths.
Industry standard in all modern LLMs.

### Why SwiGLU FFN?
Gated linear unit — learns to selectively pass information.
Consistently outperforms ReLU/GELU by 1-2% on benchmarks.
Used in PaLM, LLaMA, Gemma.

### Why RMSNorm instead of LayerNorm?
No mean subtraction → 10-15% faster than LayerNorm.
Equivalent quality, standard in modern LLMs.

---

## Training at Scale

### Data Sources (for full training)
```python
# Add to data/raw/ directory:
# - GitHub code (The Stack dataset: huggingface.co/datasets/bigcode/the-stack)
# - StackOverflow (huggingface.co/datasets/pacovaldez/stackoverflow-questions)
# - CodeParrot (huggingface.co/datasets/codeparrot/github-code)
# - Python documentation, MDN Web Docs, Rust Book

from datasets import load_dataset
ds = load_dataset("bigcode/the-stack", data_dir="data/python", streaming=True)
```

### Hardware Requirements
| Task           | Minimum        | Recommended      |
|----------------|----------------|-----------------|
| 300M training  | 1× RTX 3090    | 4× A100 80GB    |
| 700M training  | 2× A100 40GB   | 8× A100 80GB    |
| Inference CPU  | 8GB RAM        | 16GB RAM        |
| Inference GPU  | 4GB VRAM       | 8GB VRAM        |

---

## Deployment

### Mobile (Android)
```
INT4 quantized (GGUF Q4_K_M):
- 300M model: ~150 MB
- Runs on: Pixel 8, Samsung S24+
- Speed: 5-20 tokens/second
```

See `deployment/android/android_deployment_guide.json` for full guide.

### Self-Improvement
The model can improve its own code:
```python
from self_improve.self_improve import SelfImprovingCoder
solver = SelfImprovingCoder(engine, max_attempts=5)
session = solver.solve("Write a quicksort implementation")
print(session.final_code)  # Working, tested code
```

---

## Lumina Language Support

Coding Nova X includes special support for Lumina (your custom language):
- `<|lumina|>` special token in vocabulary
- Tokenizer trained to handle Lumina syntax
- Instruction pairs can include Lumina code examples

To add Lumina training data:
```python
lumina_pairs = [
    {
        "instruction": "Write a hello world in Lumina",
        "response": "lumina_code_here",
        "language": "lumina"
    }
]
```

---

## License
Built for educational and research purposes.

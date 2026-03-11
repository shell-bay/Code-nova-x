"""
Coding Nova X — Main Entry Point
==================================
Single script to run any phase of the pipeline.

Usage:
    python run.py --phase tokenizer          # Train tokenizer
    python run.py --phase pretrain           # Pretrain model
    python run.py --phase finetune           # Fine-tune model
    python run.py --phase evaluate           # Evaluate model
    python run.py --phase chat               # Interactive chat
    python run.py --phase optimize           # Quantize model
    python run.py --phase self_improve       # Self-improvement demo
    python run.py --phase all                # Run all phases sequentially
    python run.py --phase demo               # Quick demo (no training needed)
"""

import os
import sys
import argparse
import time

# Make sure imports work from anywhere
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def print_banner():
    print("""
╔══════════════════════════════════════════════════════════╗
║                                                          ║
║          🚀  CODING NOVA X  🚀                           ║
║                                                          ║
║    Transformer-Based AI Code Generation Model            ║
║    300M-1B Parameters | GPT Architecture                 ║
║    Built with PyTorch + RoPE + GQA + SwiGLU              ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
""")


def phase_tokenizer(args):
    """PHASE 2: Train tokenizer."""
    print("\n[PHASE 2] Training Tokenizer...")
    from tokenizer.nova_tokenizer import train_tokenizer
    tokenizer = train_tokenizer(
        data_dir=args.data_dir,
        output_dir="tokenizer",
        vocab_size=args.vocab_size,
    )
    print(f"[PHASE 2] ✓ Tokenizer ready! Vocab size: {tokenizer.vocab_size}")
    return tokenizer


def phase_pretrain(args):
    """PHASE 5: Pretrain the model."""
    print("\n[PHASE 5] Starting Pretraining...")
    from training.trainer import pretrain
    model, tokenizer = pretrain(
        data_dir=args.data_dir,
        checkpoint_dir="checkpoints/pretrain",
        model_size=args.model_size,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
    )
    print("[PHASE 5] ✓ Pretraining complete!")
    return model, tokenizer


def phase_finetune(args):
    """PHASE 6: Fine-tune on instruction data."""
    print("\n[PHASE 6] Starting Fine-Tuning...")
    from training.finetune import finetune
    model = finetune(
        base_model_path="checkpoints/pretrain/best_model" if os.path.exists("checkpoints/pretrain/best_model") else None,
        output_dir="checkpoints/finetuned",
        epochs=args.finetune_epochs,
        batch_size=args.batch_size,
    )
    print("[PHASE 6] ✓ Fine-tuning complete!")
    return model


def phase_evaluate(args):
    """PHASE 7: Evaluate the model."""
    print("\n[PHASE 7] Running Evaluation...")
    
    from tokenizer.nova_tokenizer import NovaTokenizer
    from configs.model_config import get_300m_config
    from model.nova_model import CodingNovaX
    from evaluation.evaluator import NovaEvaluator
    
    # Load tokenizer
    tok_path = "tokenizer/nova_tokenizer.model"
    if not os.path.exists(tok_path):
        print("[Eval] No tokenizer found, training one first...")
        tok = phase_tokenizer(args)
    else:
        tok = NovaTokenizer(tok_path)
    
    # Load or create model
    model_path = "checkpoints/finetuned"
    if os.path.exists(os.path.join(model_path, "model.pt")):
        model = CodingNovaX.from_pretrained(model_path)
    else:
        print("[Eval] No trained model found, using fresh model for structure test...")
        cfg = get_300m_config()
        cfg.vocab_size = tok.vocab_size
        cfg.use_flash_attention = False
        model = CodingNovaX(cfg)
    
    evaluator = NovaEvaluator(model, tok)
    report = evaluator.full_evaluation()
    
    print(f"\n[PHASE 7] Evaluation Results:")
    print(f"  Perplexity: {report['perplexity']:.2f}")
    print(f"  Benchmark: {report['benchmark_passed']}/{report['benchmark_total']} passed ({report['benchmark_accuracy']*100:.1f}%)")
    return report


def phase_optimize(args):
    """PHASE 9: Optimize model for deployment."""
    print("\n[PHASE 9] Optimizing Model...")
    
    from configs.model_config import get_300m_config
    from model.nova_model import CodingNovaX
    from optimization.quantize import optimize_model
    
    model_path = "checkpoints/finetuned"
    if os.path.exists(os.path.join(model_path, "model.pt")):
        model = CodingNovaX.from_pretrained(model_path)
    else:
        print("[Optimize] Using fresh model for demo...")
        cfg = get_300m_config()
        cfg.use_flash_attention = False
        model = CodingNovaX(cfg)
    
    results = optimize_model(
        model=model,
        model_config=model.config,
        output_dir="deployment",
        quantize=True,
        prune=False,
        convert_gguf=True,
    )
    print("[PHASE 9] ✓ Optimization complete!")
    return results


def phase_android(args):
    """PHASE 10: Generate Android deployment files."""
    print("\n[PHASE 10] Generating Android Deployment...")
    from deployment.android.android_inference import create_android_project_guide
    guide_path = create_android_project_guide()
    print(f"[PHASE 10] ✓ Android guide created: {guide_path}")


def phase_chat(args):
    """Interactive chat mode."""
    print("\n[Chat] Starting interactive chat...")
    
    from tokenizer.nova_tokenizer import NovaTokenizer, train_tokenizer
    from configs.model_config import get_300m_config
    from model.nova_model import CodingNovaX
    from inference.engine import NovaInferenceEngine, GenerationConfig
    
    # Load tokenizer
    tok_path = "tokenizer/nova_tokenizer.model"
    if os.path.exists(tok_path):
        tokenizer = NovaTokenizer(tok_path)
    else:
        print("[Chat] Training tokenizer first...")
        tokenizer = train_tokenizer(vocab_size=8000)
    
    # Load model
    model_path = "checkpoints/finetuned"
    if os.path.exists(os.path.join(model_path, "model.pt")):
        model = CodingNovaX.from_pretrained(model_path)
    else:
        print("[Chat] Using untrained model (run --phase pretrain first for good results)")
        cfg = get_300m_config()
        cfg.vocab_size = tokenizer.vocab_size
        cfg.use_flash_attention = False
        model = CodingNovaX(cfg)
    
    engine = NovaInferenceEngine(model=model, tokenizer=tokenizer)
    gen_cfg = GenerationConfig(max_new_tokens=256, temperature=0.7)
    
    print("\n" + "="*60)
    print("  Coding Nova X — Chat Mode")
    print("  Type your coding question (or 'quit' to exit)")
    print("="*60)
    
    while True:
        try:
            user_input = input("\n👤 You: ").strip()
            if not user_input or user_input.lower() in ('quit', 'exit', 'q'):
                print("Goodbye!")
                break
            
            print("🤖 Nova: ", end="", flush=True)
            response = engine.generate(user_input, config=gen_cfg, stream=True)
        except KeyboardInterrupt:
            print("\nExiting...")
            break


def phase_self_improve(args):
    """PHASE 11: Self-improving loop demo."""
    print("\n[PHASE 11] Self-Improvement Demo...")
    
    from tokenizer.nova_tokenizer import NovaTokenizer, train_tokenizer
    from configs.model_config import get_300m_config
    from model.nova_model import CodingNovaX
    from inference.engine import NovaInferenceEngine
    from self_improve.self_improve import SelfImprovingCoder, SELF_IMPROVEMENT_TASKS
    
    tok_path = "tokenizer/nova_tokenizer.model"
    tokenizer = NovaTokenizer(tok_path) if os.path.exists(tok_path) else train_tokenizer(vocab_size=8000)
    
    model_path = "checkpoints/finetuned"
    if os.path.exists(os.path.join(model_path, "model.pt")):
        model = CodingNovaX.from_pretrained(model_path)
    else:
        cfg = get_300m_config()
        cfg.vocab_size = tokenizer.vocab_size
        cfg.use_flash_attention = False
        model = CodingNovaX(cfg)
    
    engine = NovaInferenceEngine(model=model, tokenizer=tokenizer)
    solver = SelfImprovingCoder(engine, max_attempts=3)
    
    # Run first test task
    task = SELF_IMPROVEMENT_TASKS[0]
    session = solver.solve(task["task"], task["test_code"], verbose=True)
    
    print(f"\n[PHASE 11] Session complete:")
    print(f"  Task solved: {session.success}")
    print(f"  Attempts used: {session.total_attempts}")
    return session


def phase_demo(args):
    """Demo all components without training."""
    print("\n[DEMO] Running full system demo...")
    
    # 1. Config
    print("\n--- Model Configuration ---")
    from configs.model_config import get_300m_config, get_700m_config
    cfg_300m = get_300m_config()
    cfg_700m = get_700m_config()
    print(f"300M config: {cfg_300m.estimate_parameters()/1e6:.1f}M parameters")
    print(f"700M config: {cfg_700m.estimate_parameters()/1e6:.1f}M parameters")
    
    # 2. Tokenizer
    print("\n--- Tokenizer ---")
    from tokenizer.nova_tokenizer import NovaTokenizer, train_tokenizer
    tok_path = "tokenizer/nova_tokenizer.model"
    if os.path.exists(tok_path):
        tok = NovaTokenizer(tok_path)
    else:
        tok = train_tokenizer(vocab_size=4000)
    
    test_text = "def hello(): return 'Hello, Coding Nova X!'"
    ids = tok.encode(test_text)
    decoded = tok.decode(ids)
    print(f"  Input:  {test_text}")
    print(f"  Tokens: {ids[:10]}...")
    print(f"  Decoded: {decoded}")
    
    # 3. Model
    print("\n--- Model Architecture ---")
    import torch
    from model.nova_model import CodingNovaX
    cfg = get_300m_config()
    cfg.vocab_size = tok.vocab_size
    cfg.use_flash_attention = False
    cfg.num_layers = 4  # Tiny for demo
    model = CodingNovaX(cfg)
    print(f"  Parameters: {model.count_parameters()/1e6:.1f}M")
    
    # Forward pass test
    ids_tensor = torch.tensor([ids[:32]], dtype=torch.long)
    with torch.no_grad():
        out = model(ids_tensor, labels=ids_tensor)
    print(f"  Forward pass ✓ | Loss: {out['loss'].item():.4f}")
    
    # 4. Dataset
    print("\n--- Dataset Pipeline ---")
    from dataset.data_pipeline import get_builtin_instruction_data, DeduplicatedDataset
    pairs = get_builtin_instruction_data()
    dedup = DeduplicatedDataset()
    filtered = dedup.filter_dataset([p['instruction'] for p in pairs])
    print(f"  Built-in instruction pairs: {len(pairs)}")
    print(f"  After dedup: {len(filtered)}")
    
    # 5. Android guide
    print("\n--- Android Deployment ---")
    from deployment.android.android_inference import create_android_project_guide
    create_android_project_guide()
    
    print("\n" + "="*60)
    print("  ✓ DEMO COMPLETE — All systems functional!")
    print("  Next step: python run.py --phase pretrain")
    print("="*60)


def phase_all(args):
    """Run all phases sequentially."""
    print("\n[ALL] Running complete Coding Nova X pipeline...")
    
    phase_tokenizer(args)
    phase_pretrain(args)
    phase_finetune(args)
    phase_evaluate(args)
    phase_optimize(args)
    phase_android(args)
    
    print("\n" + "="*60)
    print("  ✓ ALL PHASES COMPLETE!")
    print("  Run: python run.py --phase chat")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Coding Nova X — AI Code Generation Model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py --phase demo             # Quick demo
  python run.py --phase tokenizer        # Train tokenizer
  python run.py --phase pretrain --max_steps 1000
  python run.py --phase finetune --finetune_epochs 3
  python run.py --phase evaluate
  python run.py --phase chat
  python run.py --phase optimize
  python run.py --phase self_improve
        """
    )
    
    parser.add_argument("--phase", required=True, choices=[
        "tokenizer", "pretrain", "finetune", "evaluate",
        "chat", "optimize", "android", "self_improve", "all", "demo"
    ], help="Which phase to run")
    
    parser.add_argument("--model_size", default="300m", choices=["300m", "700m", "1b"])
    parser.add_argument("--data_dir", default="data/processed")
    parser.add_argument("--vocab_size", type=int, default=32000)
    parser.add_argument("--max_steps", type=int, default=500000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--finetune_epochs", type=int, default=3)
    
    args = parser.parse_args()
    
    print_banner()
    print(f"Phase: {args.phase.upper()}")
    print(f"Model size: {args.model_size}")
    
    phase_map = {
        "tokenizer": phase_tokenizer,
        "pretrain": phase_pretrain,
        "finetune": phase_finetune,
        "evaluate": phase_evaluate,
        "chat": phase_chat,
        "optimize": phase_optimize,
        "android": phase_android,
        "self_improve": phase_self_improve,
        "all": phase_all,
        "demo": phase_demo,
    }
    
    t_start = time.time()
    result = phase_map[args.phase](args)
    elapsed = time.time() - t_start
    
    print(f"\n[Done] Phase '{args.phase}' completed in {elapsed:.1f}s")
    return result


if __name__ == "__main__":
    main()

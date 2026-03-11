"""
Coding Nova X — Inference Engine
==================================
Fast, user-friendly inference for code generation.

Features:
- KV caching for fast autoregressive generation
- Temperature, top-k, top-p sampling
- Streaming output (print tokens as generated)
- Multi-turn conversation support
- Code extraction from output
"""

import os
import sys
import re
import time
import torch
import torch.nn.functional as F
from typing import Optional, List, Generator

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


SYSTEM_PROMPT = """You are Coding Nova X, an expert AI coding assistant. 
You write clean, efficient, well-documented code and explain concepts clearly.
You support Python, C++, JavaScript, Java, Rust, and more.
When asked to write code, always include comments explaining key steps."""


class GenerationConfig:
    """Settings for text generation."""
    def __init__(
        self,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        do_sample: bool = True,
        repetition_penalty: float = 1.1,
    ):
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.do_sample = do_sample
        self.repetition_penalty = repetition_penalty


class NovaInferenceEngine:
    """
    Inference engine for Coding Nova X.
    
    Handles model loading, prompt formatting, generation,
    and post-processing of outputs.
    """

    def __init__(
        self,
        model=None,
        tokenizer=None,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        dtype: str = "float32",
    ):
        """
        Initialize engine. Either pass model+tokenizer directly,
        or provide model_path to load from disk.
        """
        # Device setup
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = torch.device(device)
        
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        self.dtype = dtype_map.get(dtype, torch.float32)
        
        # Load model
        if model is not None:
            self.model = model.to(self.device)
        elif model_path:
            self.model = self._load_model(model_path)
        else:
            raise ValueError("Provide model or model_path")
        
        self.model.eval()
        self.tokenizer = tokenizer
        
        # Conversation history for multi-turn chat
        self.conversation_history: List[dict] = []
        
        print(f"[Engine] Coding Nova X ready on {self.device}")
        print(f"[Engine] Model params: {sum(p.numel() for p in self.model.parameters())/1e6:.1f}M")

    def _load_model(self, path: str):
        from model.nova_model import CodingNovaX
        model = CodingNovaX.from_pretrained(path)
        return model.to(self.device).to(self.dtype)

    def _apply_repetition_penalty(
        self, logits: torch.Tensor, input_ids: torch.Tensor, penalty: float
    ) -> torch.Tensor:
        """
        Reduce probability of tokens already generated.
        Prevents the model from endlessly repeating itself.
        """
        if penalty == 1.0:
            return logits
        for token_id in set(input_ids[0].tolist()):
            if logits[0, token_id] < 0:
                logits[0, token_id] *= penalty
            else:
                logits[0, token_id] /= penalty
        return logits

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        stream: bool = False,
    ) -> str:
        """
        Generate a response for a prompt.
        
        Args:
            prompt: The user's instruction or question
            config: Generation settings
            stream: If True, prints tokens as generated
        
        Returns:
            Generated text string
        """
        if config is None:
            config = GenerationConfig()
        
        if self.tokenizer is None:
            raise RuntimeError("No tokenizer loaded!")
        
        # Format the prompt
        formatted = self.tokenizer.format_instruction(
            instruction=prompt,
            system=SYSTEM_PROMPT,
        )
        
        ids = self.tokenizer.encode(formatted, add_bos=True, add_eos=False)
        input_tensor = torch.tensor([ids], dtype=torch.long, device=self.device)
        
        generated_ids = []
        past_kv = None
        
        t_start = time.time()
        
        for step in range(config.max_new_tokens):
            # Use KV cache — only process the new token after first step
            if past_kv is not None:
                curr_input = torch.tensor([[generated_ids[-1]]], dtype=torch.long, device=self.device)
            else:
                curr_input = input_tensor
            
            outputs = self.model(curr_input, past_key_values=past_kv, use_cache=True)
            logits = outputs["logits"][:, -1, :].float()   # [1, vocab]
            past_kv = outputs["past_key_values"]
            
            # Repetition penalty
            all_ids = torch.tensor([ids + generated_ids], device=self.device)
            logits = self._apply_repetition_penalty(logits, all_ids, config.repetition_penalty)
            
            # Temperature
            if config.temperature != 1.0:
                logits = logits / config.temperature
            
            # Top-K
            if config.top_k > 0:
                k = min(config.top_k, logits.size(-1))
                topk_vals, _ = torch.topk(logits, k)
                logits[logits < topk_vals[:, -1:]] = float("-inf")
            
            # Top-P
            if config.top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                remove_mask = cum_probs - F.softmax(sorted_logits, dim=-1) > config.top_p
                sorted_logits[remove_mask] = float("-inf")
                logits = torch.zeros_like(logits).scatter_(1, sorted_idx, sorted_logits)
            
            # Sample
            if config.do_sample:
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
            else:
                next_token = logits.argmax(dim=-1).item()
            
            # Stop at EOS
            if next_token == self.tokenizer.eos_id:
                break
            
            generated_ids.append(next_token)
            
            # Stream output
            if stream:
                token_text = self.tokenizer.decode([next_token], skip_special=True)
                print(token_text, end="", flush=True)
        
        if stream:
            print()  # New line after streaming
        
        elapsed = time.time() - t_start
        tokens_per_sec = len(generated_ids) / max(elapsed, 0.001)
        
        output_text = self.tokenizer.decode(generated_ids, skip_special=True)
        
        print(f"\n[Engine] Generated {len(generated_ids)} tokens in {elapsed:.2f}s ({tokens_per_sec:.1f} tok/s)")
        return output_text

    def chat(self, message: str, **kwargs) -> str:
        """
        Multi-turn conversation interface.
        Remembers previous messages in the conversation.
        """
        # Build full conversation context
        history_text = ""
        for turn in self.conversation_history[-6:]:  # Keep last 6 turns
            history_text += f"Previous: {turn['role']}: {turn['content'][:200]}\n"
        
        full_prompt = history_text + message if history_text else message
        response = self.generate(full_prompt, **kwargs)
        
        # Save to history
        self.conversation_history.append({"role": "human", "content": message})
        self.conversation_history.append({"role": "assistant", "content": response})
        
        return response

    def reset_chat(self):
        """Clear conversation history."""
        self.conversation_history = []
        print("[Engine] Conversation reset.")

    def extract_code_blocks(self, text: str) -> List[str]:
        """
        Extract code blocks from generated text.
        Looks for ```...``` markdown blocks or indented code.
        """
        # Markdown code blocks
        pattern = r"```(?:\w+)?\n?(.*?)```"
        blocks = re.findall(pattern, text, re.DOTALL)
        
        if not blocks:
            # Try to find Python-looking code
            lines = text.split('\n')
            code_lines = []
            in_code = False
            for line in lines:
                if line.startswith('def ') or line.startswith('class '):
                    in_code = True
                if in_code:
                    code_lines.append(line)
            if code_lines:
                blocks = ['\n'.join(code_lines)]
        
        return [b.strip() for b in blocks if b.strip()]

    def code_complete(self, code_prefix: str, **kwargs) -> str:
        """Complete a partial code snippet."""
        prompt = f"Complete the following code:\n\n```python\n{code_prefix}\n```"
        return self.generate(prompt, **kwargs)

    def debug_code(self, code: str, error_message: str = "", **kwargs) -> str:
        """Debug broken code."""
        prompt = f"Debug and fix this code:\n\n```python\n{code}\n```"
        if error_message:
            prompt += f"\n\nError message:\n{error_message}"
        return self.generate(prompt, **kwargs)

    def explain_code(self, code: str, **kwargs) -> str:
        """Explain what code does."""
        prompt = f"Explain what this code does step by step:\n\n```python\n{code}\n```"
        return self.generate(prompt, **kwargs)


# ── Interactive CLI ────────────────────────────────────────────────────────────

def interactive_cli(model_path: str = "checkpoints/finetuned"):
    """Run interactive command-line interface."""
    from tokenizer.nova_tokenizer import NovaTokenizer
    
    print("\n" + "=" * 60)
    print("  🚀 Coding Nova X — Interactive Mode")
    print("=" * 60)
    print("Commands: /reset (clear history), /debug <code>, /explain <code>, /quit")
    print("-" * 60 + "\n")
    
    tokenizer = NovaTokenizer("tokenizer/nova_tokenizer.model")
    engine = NovaInferenceEngine(
        tokenizer=tokenizer,
        model_path=model_path if os.path.exists(os.path.join(model_path, "model.pt")) else None,
    )
    
    gen_cfg = GenerationConfig(
        max_new_tokens=512,
        temperature=0.7,
        top_k=50,
        top_p=0.9,
    )
    
    while True:
        try:
            user_input = input("\n👤 You: ").strip()
            
            if not user_input:
                continue
            elif user_input.lower() == "/quit":
                print("Goodbye! 👋")
                break
            elif user_input.lower() == "/reset":
                engine.reset_chat()
            else:
                print("\n🤖 Coding Nova X: ", end="")
                response = engine.chat(user_input, config=gen_cfg, stream=True)
        
        except KeyboardInterrupt:
            print("\n\nInterrupted. Type /quit to exit.")
        except Exception as e:
            print(f"\n[Error] {e}")


if __name__ == "__main__":
    print("Coding Nova X Inference Engine ready.")
    print("To use: from inference.engine import NovaInferenceEngine")

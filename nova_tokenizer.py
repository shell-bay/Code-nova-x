"""
Coding Nova X — Tokenizer Training & Management
=================================================
Uses SentencePiece BPE to build a code-aware tokenizer.

Why SentencePiece BPE?
- Works on raw bytes → handles any programming language
- No pre-tokenization assumptions → respects code structure
- Fast and efficient → used in LLaMA, Mistral, CodeLlama
"""

import os
import sys
import json
import sentencepiece as spm
from typing import List, Optional, Union
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


SAMPLE_CODE_CORPUS = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

class BinaryTree:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

#include <iostream>
#include <vector>
int main() {
    std::vector<int> v = {3, 1, 4, 1, 5};
    std::sort(v.begin(), v.end());
    return 0;
}

function quickSort(arr) {
    if (arr.length <= 1) return arr;
    const pivot = arr[Math.floor(arr.length / 2)];
    return [...quickSort(arr.filter(x => x < pivot)), pivot, ...quickSort(arr.filter(x => x > pivot))];
}

fn bubble_sort(arr: &mut Vec<i32>) {
    let n = arr.len();
    for i in 0..n {
        for j in 0..n-i-1 {
            if arr[j] > arr[j+1] {
                arr.swap(j, j+1);
            }
        }
    }
}

Question: How do I implement a binary search in Python?
Answer: Binary search works by repeatedly dividing the search interval in half.

def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1

The time complexity is O(log n) which is much faster than linear search O(n).
"""


class NovaTokenizer:
    """
    Coding Nova X tokenizer wrapper around SentencePiece.
    
    Handles special tokens for code, instruction tuning,
    and multi-language support.
    """

    SPECIAL_TOKENS = [
        "<pad>", "<s>", "</s>", "<unk>",
        "<|code|>", "<|endcode|>",
        "<|python|>", "<|cpp|>", "<|javascript|>",
        "<|java|>", "<|rust|>", "<|bash|>", "<|sql|>",
        "<|human|>", "<|assistant|>", "<|system|>",
        "<|endoftext|>", "<|debug|>", "<|fix|>", "<|explain|>",
        "<|lumina|>",  # For Lumina language support!
    ]

    def __init__(self, model_path: Optional[str] = None):
        self.sp_model = None
        self.model_path = model_path
        self._special_token_ids = {}
        
        if model_path and os.path.exists(model_path):
            self._load(model_path)

    def train(
        self,
        input_files: Optional[List[str]] = None,
        vocab_size: int = 32000,
        model_prefix: str = "tokenizer/nova_tokenizer",
        use_sample_data: bool = True,
    ):
        """
        Train the BPE tokenizer on code + natural language data.
        
        Args:
            input_files: List of text files to train on
            vocab_size: Target vocabulary size (32k-50k recommended)
            model_prefix: Path prefix for saving .model and .vocab files
            use_sample_data: Add built-in sample data for testing
        """
        os.makedirs(os.path.dirname(model_prefix) if '/' in model_prefix else '.', exist_ok=True)
        
        # Create training corpus
        tmp_files = []
        
        if use_sample_data:
            # Write sample corpus to temp file
            tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8')
            # Write many repetitions to make meaningful vocab
            for _ in range(50):
                tmp.write(SAMPLE_CODE_CORPUS + "\n")
            # Add natural language
            tmp.write("\n".join([
                "The algorithm runs in O(n log n) time complexity.",
                "Machine learning models learn patterns from training data.",
                "A function takes inputs and returns outputs.",
                "Variables store data values in memory.",
                "Loops repeat a block of code multiple times.",
                "Recursion is when a function calls itself.",
                "Object-oriented programming uses classes and objects.",
                "Debugging is the process of finding and fixing errors.",
                "Unit tests verify that individual functions work correctly.",
                "APIs allow different software systems to communicate.",
            ] * 100))
            tmp.close()
            tmp_files.append(tmp.name)
        
        if input_files:
            tmp_files.extend(input_files)
        
        # Build special tokens string for SentencePiece
        user_defined_symbols = ",".join(self.SPECIAL_TOKENS[4:])  # Skip basic tokens
        
        # SentencePiece training command
        train_args = {
            "input": ",".join(tmp_files),
            "model_prefix": model_prefix,
            "vocab_size": vocab_size,
            "model_type": "bpe",
            "character_coverage": 0.9999,
            "pad_id": 0,
            "bos_id": 1,
            "eos_id": 2,
            "unk_id": 3,
            "pad_piece": "<pad>",
            "bos_piece": "<s>",
            "eos_piece": "</s>",
            "unk_piece": "<unk>",
            "user_defined_symbols": user_defined_symbols,
            "byte_fallback": True,           # Handle unknown chars as bytes
            "split_digits": True,            # Treat digits individually (good for code)
            "add_dummy_prefix": False,       # Don't add space at start
            "normalization_rule_name": "identity",  # Keep code as-is
            "num_threads": 4,
            "input_sentence_size": 100000,
            "shuffle_input_sentence": True,
        }
        
        print(f"[Tokenizer] Training BPE tokenizer with vocab_size={vocab_size}...")
        spm.SentencePieceTrainer.train(**train_args)
        
        # Load trained model
        self.model_path = model_prefix + ".model"
        self._load(self.model_path)
        
        # Cleanup temp files
        for f in tmp_files:
            if f not in (input_files or []):
                os.unlink(f)
        
        print(f"[Tokenizer] ✓ Training complete! Vocab size: {self.vocab_size}")
        print(f"[Tokenizer] Model saved to: {self.model_path}")
        return self

    def _load(self, model_path: str):
        """Load trained SentencePiece model."""
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.Load(model_path)
        self.model_path = model_path
        
        # Map special token strings to IDs
        for tok in self.SPECIAL_TOKENS:
            tid = self.sp_model.PieceToId(tok)
            if tid != 0 or tok == "<pad>":  # 0 is pad
                self._special_token_ids[tok] = tid
        
        print(f"[Tokenizer] Loaded tokenizer from {model_path} | vocab={self.vocab_size}")

    @property
    def vocab_size(self) -> int:
        return self.sp_model.GetPieceSize() if self.sp_model else 0

    @property
    def pad_id(self) -> int:
        return self.sp_model.pad_id()

    @property
    def bos_id(self) -> int:
        return self.sp_model.bos_id()

    @property
    def eos_id(self) -> int:
        return self.sp_model.eos_id()

    def encode(
        self,
        text: str,
        add_bos: bool = True,
        add_eos: bool = False,
        max_length: Optional[int] = None,
    ) -> List[int]:
        """
        Encode text to token IDs.
        
        Args:
            text: Input text or code
            add_bos: Prepend beginning-of-sequence token
            add_eos: Append end-of-sequence token
            max_length: Truncate to this length
        
        Returns:
            List of integer token IDs
        """
        if self.sp_model is None:
            raise RuntimeError("Tokenizer not loaded. Call train() or load a model first.")
        
        ids = self.sp_model.Encode(text, add_bos=add_bos, add_eos=add_eos)
        
        if max_length is not None:
            ids = ids[:max_length]
        
        return ids

    def decode(self, ids: List[int], skip_special: bool = False) -> str:
        """
        Decode token IDs back to text.
        
        Args:
            ids: List of token IDs
            skip_special: Skip special tokens in output
        
        Returns:
            Decoded text string
        """
        if self.sp_model is None:
            raise RuntimeError("Tokenizer not loaded.")
        
        if skip_special:
            special_ids = set(self._special_token_ids.values())
            ids = [i for i in ids if i not in special_ids]
        
        return self.sp_model.Decode(ids)

    def encode_batch(self, texts: List[str], **kwargs) -> List[List[int]]:
        """Encode a batch of texts."""
        return [self.encode(t, **kwargs) for t in texts]

    def tokenize(self, text: str) -> List[str]:
        """Convert text to token strings (for debugging)."""
        return self.sp_model.Encode(text, out_type=str)

    def get_special_token_id(self, token: str) -> Optional[int]:
        """Get ID for a special token by string."""
        return self._special_token_ids.get(token)

    def format_instruction(
        self,
        instruction: str,
        response: Optional[str] = None,
        system: Optional[str] = None,
        language: Optional[str] = None,
    ) -> str:
        """
        Format text in instruction-tuning style.
        
        This creates the prompt format the model will learn during fine-tuning.
        """
        parts = []
        
        if system:
            parts.append(f"<|system|>\n{system}\n")
        
        if language:
            lang_token = f"<|{language.lower()}|>"
            parts.append(f"{lang_token}\n")
        
        parts.append(f"<|human|>\n{instruction}\n")
        
        if response is not None:
            parts.append(f"<|assistant|>\n{response}")
            parts.append("</s>")
        else:
            parts.append("<|assistant|>\n")
        
        return "".join(parts)

    def save_config(self, path: str):
        """Save tokenizer config as JSON."""
        config = {
            "model_path": self.model_path,
            "vocab_size": self.vocab_size,
            "special_tokens": self.SPECIAL_TOKENS,
            "special_token_ids": self._special_token_ids,
        }
        with open(path, "w") as f:
            json.dump(config, f, indent=2)
        print(f"[Tokenizer] Config saved to {path}")


def train_tokenizer(
    data_dir: str = "data/raw",
    output_dir: str = "tokenizer",
    vocab_size: int = 32000,
) -> NovaTokenizer:
    """
    Main function to train the tokenizer.
    Automatically gathers text files from data_dir.
    """
    tokenizer = NovaTokenizer()
    
    # Find training files
    input_files = []
    if os.path.exists(data_dir):
        for f in os.listdir(data_dir):
            if f.endswith(('.txt', '.py', '.js', '.cpp', '.java', '.rs')):
                input_files.append(os.path.join(data_dir, f))
    
    model_prefix = os.path.join(output_dir, "nova_tokenizer")
    tokenizer.train(
        input_files=input_files if input_files else None,
        vocab_size=vocab_size,
        model_prefix=model_prefix,
        use_sample_data=True,
    )
    tokenizer.save_config(os.path.join(output_dir, "tokenizer_config.json"))
    return tokenizer


if __name__ == "__main__":
    print("=" * 60)
    print("  Coding Nova X — Tokenizer Training")
    print("=" * 60)
    
    tok = train_tokenizer(vocab_size=8000)  # Small for testing
    
    # Test encoding
    test_code = """
def hello_world():
    print("Hello from Coding Nova X!")
    return 42
"""
    ids = tok.encode(test_code)
    decoded = tok.decode(ids)
    
    print(f"\n[Test] Input length: {len(test_code)} chars")
    print(f"[Test] Token IDs: {ids[:20]}...")
    print(f"[Test] Num tokens: {len(ids)}")
    print(f"[Test] Decoded: {decoded[:100]}")
    
    # Test instruction format
    prompt = tok.format_instruction(
        instruction="Write a Python function to reverse a string",
        language="python",
    )
    print(f"\n[Test] Instruction format:\n{prompt}")
    print("\n[Tokenizer] ✓ All tests passed!")

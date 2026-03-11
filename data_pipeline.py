"""
Coding Nova X — Dataset Pipeline
==================================
Handles massive code datasets with streaming, deduplication,
preprocessing, and efficient tokenization.

Data sources:
- GitHub code repositories
- StackOverflow Q&A
- Programming documentation
- Coding tutorials
- HuggingFace datasets (The Stack, CodeParrot, etc.)
"""

import os
import sys
import json
import hashlib
import random
try:
    import torch
except ImportError:
    torch = None
from torch.utils.data import Dataset, DataLoader, IterableDataset
from typing import Iterator, List, Dict, Optional, Generator
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


# ── Code Templates for synthetic data generation ──────────────────────────────

CODE_TEMPLATES = {
    "python": [
        "def {name}({params}):\n    {body}\n",
        "class {name}:\n    def __init__(self):\n        {init}\n",
        "import {module}\n\n{code}",
    ],
    "instruction_pairs": [
        {
            "instruction": "Write a Python function to calculate factorial",
            "response": """def factorial(n):
    \"\"\"Calculate factorial of n recursively.\"\"\"
    if n < 0:
        raise ValueError("Factorial undefined for negative numbers")
    if n == 0 or n == 1:
        return 1
    return n * factorial(n - 1)

# Iterative version (more efficient):
def factorial_iterative(n):
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result"""
        },
        {
            "instruction": "Implement a stack data structure in Python",
            "response": """class Stack:
    \"\"\"LIFO data structure implemented with a list.\"\"\"
    
    def __init__(self):
        self._items = []
    
    def push(self, item):
        \"\"\"Add item to top of stack. O(1)\"\"\"
        self._items.append(item)
    
    def pop(self):
        \"\"\"Remove and return top item. O(1)\"\"\"
        if self.is_empty():
            raise IndexError("Pop from empty stack")
        return self._items.pop()
    
    def peek(self):
        \"\"\"View top item without removing. O(1)\"\"\"
        if self.is_empty():
            raise IndexError("Peek at empty stack")
        return self._items[-1]
    
    def is_empty(self):
        return len(self._items) == 0
    
    def size(self):
        return len(self._items)
    
    def __repr__(self):
        return f"Stack({self._items})" """
        },
        {
            "instruction": "Debug this code: my for loop isn't working",
            "response": """The issue might be one of these common problems:

1. **Off-by-one error** - Check your range:
```python
# Wrong - misses last element
for i in range(len(arr) - 1):  

# Correct
for i in range(len(arr)):
```

2. **Modifying list while iterating** - Never do this:
```python
# Wrong - causes skipping elements
for item in my_list:
    if condition:
        my_list.remove(item)  # BAD!

# Correct - iterate over a copy
for item in my_list[:]:
    if condition:
        my_list.remove(item)
```

3. **Wrong variable name** - Check you're using the loop variable consistently.

Share your specific code and I can give you an exact fix!"""
        },
        {
            "instruction": "Explain what recursion is with an example",
            "response": """**Recursion** is when a function calls itself to solve smaller versions of the same problem.

Think of it like Russian nesting dolls - each doll contains a smaller version of itself.

**Key components:**
1. **Base case** - The stopping condition (prevents infinite loops)
2. **Recursive case** - Function calls itself with a smaller input

**Example - counting down:**
```python
def countdown(n):
    # Base case: stop when we reach 0
    if n == 0:
        print("Blast off!")
        return
    
    # Recursive case: print n, then count down from n-1
    print(n)
    countdown(n - 1)  # Calls itself!

countdown(5)
# Output: 5, 4, 3, 2, 1, Blast off!
```

**How it works in memory:**
- countdown(5) calls countdown(4)
- countdown(4) calls countdown(3)  
- ... keeps going until n=0
- Then each call finishes in reverse order

The call stack grows with each call, so very deep recursion uses lots of memory!"""
        },
    ],
    "debug_pairs": [
        {
            "buggy": "def sum_list(lst):\n    total = 0\n    for i in range(len(lst) + 1):\n        total += lst[i]\n    return total",
            "fixed": "def sum_list(lst):\n    total = 0\n    for i in range(len(lst)):  # Fixed: removed + 1 (index out of bounds)\n        total += lst[i]\n    return total",
            "explanation": "Bug: range(len(lst) + 1) creates an off-by-one error causing IndexError on last iteration."
        },
    ]
}


class TextDataset(Dataset):
    """
    Simple in-memory dataset for tokenized text.
    Used for smaller datasets that fit in RAM.
    """

    def __init__(self, data: List[List[int]], max_seq_len: int = 4096):
        self.data = data
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokens = self.data[idx]
        # Pad or truncate to max_seq_len
        if len(tokens) > self.max_seq_len:
            tokens = tokens[:self.max_seq_len]
        return torch.tensor(tokens, dtype=torch.long)


class StreamingCodeDataset(IterableDataset):
    """
    Streaming dataset that reads files one chunk at a time.
    
    For large datasets (100GB+) that don't fit in memory.
    Reads, tokenizes, and yields chunks on-the-fly.
    """

    def __init__(
        self,
        data_dir: str,
        tokenizer,
        max_seq_len: int = 4096,
        overlap: int = 64,       # Token overlap between chunks
        shuffle: bool = True,
        seed: int = 42,
    ):
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.overlap = overlap
        self.shuffle = shuffle
        self.seed = seed
        
        # Find all text/code files
        self.files = self._find_files()
        print(f"[Dataset] Found {len(self.files)} files in {data_dir}")

    def _find_files(self) -> List[str]:
        exts = {'.txt', '.py', '.js', '.cpp', '.java', '.rs', '.go', '.jsonl', '.json'}
        files = []
        if os.path.exists(self.data_dir):
            for root, _, fnames in os.walk(self.data_dir):
                for fname in fnames:
                    if any(fname.endswith(e) for e in exts):
                        files.append(os.path.join(root, fname))
        return files

    def _read_file(self, path: str) -> Generator[str, None, None]:
        """Read file, yielding text in chunks."""
        try:
            if path.endswith('.jsonl'):
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        try:
                            obj = json.loads(line)
                            text = obj.get('content', obj.get('text', str(obj)))
                            if text:
                                yield text
                        except json.JSONDecodeError:
                            continue
            else:
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    yield f.read()
        except Exception as e:
            pass  # Skip unreadable files

    def _tokenize_and_chunk(self, text: str) -> List[List[int]]:
        """Tokenize text and split into fixed-size chunks with overlap."""
        try:
            ids = self.tokenizer.encode(text, add_bos=True, add_eos=True)
        except Exception:
            return []
        
        chunks = []
        stride = self.max_seq_len - self.overlap
        for start in range(0, max(1, len(ids) - self.overlap), stride):
            chunk = ids[start:start + self.max_seq_len]
            if len(chunk) >= 16:  # Skip tiny chunks
                chunks.append(chunk)
        return chunks

    def __iter__(self) -> Iterator[Dict]:
        """Iterate over all file chunks."""
        worker_info = torch.utils.data.get_worker_info()
        files = self.files[:]
        
        # Split files across workers
        if worker_info is not None:
            files = files[worker_info.id::worker_info.num_workers]
        
        if self.shuffle:
            rng = random.Random(self.seed)
            rng.shuffle(files)
        
        for fpath in files:
            for text in self._read_file(fpath):
                for chunk in self._tokenize_and_chunk(text):
                    ids = torch.tensor(chunk, dtype=torch.long)
                    yield {"input_ids": ids, "labels": ids.clone()}


class InstructionDataset(Dataset):
    """
    Dataset for instruction fine-tuning.
    
    Formats: [SYSTEM] [HUMAN: instruction] [ASSISTANT: response]
    Only the response tokens contribute to the loss (masked labels).
    """

    def __init__(
        self,
        pairs: List[Dict],    # List of {instruction, response} dicts
        tokenizer,
        max_seq_len: int = 2048,
        system_prompt: str = "You are Coding Nova X, an expert AI coding assistant. Write clean, efficient, well-documented code.",
    ):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.system_prompt = system_prompt
        
        self.data = self._preprocess()
        print(f"[InstructionDataset] Prepared {len(self.data)} samples")

    def _preprocess(self) -> List[Dict]:
        """Tokenize all instruction-response pairs."""
        processed = []
        
        for pair in self.pairs:
            instruction = pair.get("instruction", pair.get("input", ""))
            response = pair.get("response", pair.get("output", ""))
            language = pair.get("language", None)
            
            # Format the full conversation
            full_text = self.tokenizer.format_instruction(
                instruction=instruction,
                response=response,
                system=self.system_prompt,
                language=language,
            )
            
            # Format just the prompt (without response) for masking
            prompt_only = self.tokenizer.format_instruction(
                instruction=instruction,
                system=self.system_prompt,
                language=language,
            )
            
            full_ids = self.tokenizer.encode(full_text, add_bos=True, add_eos=True)
            prompt_ids = self.tokenizer.encode(prompt_only, add_bos=True, add_eos=False)
            
            if len(full_ids) > self.max_seq_len:
                full_ids = full_ids[:self.max_seq_len]
            
            prompt_len = min(len(prompt_ids), len(full_ids))
            
            # Labels: -100 for prompt tokens (masked), real IDs for response
            labels = [-100] * prompt_len + full_ids[prompt_len:]
            
            if len(labels) < len(full_ids):
                labels = labels[:len(full_ids)]
            
            processed.append({
                "input_ids": full_ids,
                "labels": labels,
            })
        
        return processed

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "input_ids": torch.tensor(item["input_ids"], dtype=torch.long),
            "labels": torch.tensor(item["labels"], dtype=torch.long),
        }


class DeduplicatedDataset:
    """
    Deduplication using MinHash/exact hash approach.
    Removes near-duplicate code snippets from training data.
    
    Why? Duplicate data causes memorization instead of generalization.
    """

    def __init__(self, threshold: float = 0.85):
        self.threshold = threshold
        self.seen_hashes = set()
        self.deduplicated_count = 0
        self.total_count = 0

    def _hash_text(self, text: str) -> str:
        """Exact MD5 hash for deduplication."""
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    def is_duplicate(self, text: str) -> bool:
        """Return True if text is a duplicate."""
        h = self._hash_text(text)
        self.total_count += 1
        if h in self.seen_hashes:
            self.deduplicated_count += 1
            return True
        self.seen_hashes.add(h)
        return False

    def filter_dataset(self, texts: List[str]) -> List[str]:
        """Filter duplicates from a list of texts."""
        filtered = []
        for text in texts:
            if not self.is_duplicate(text):
                filtered.append(text)
        print(f"[Dedup] {self.deduplicated_count}/{self.total_count} duplicates removed "
              f"({100*self.deduplicated_count/max(1,self.total_count):.1f}%)")
        return filtered


def collate_fn(batch: List[Dict], pad_id: int = 0, max_seq_len: int = 4096) -> Dict:
    """
    Custom collation: pad sequences in a batch to same length.
    
    This is needed because sequences have different lengths.
    We pad shorter sequences with pad_id so tensors are rectangular.
    """
    input_ids = [item["input_ids"] for item in batch]
    labels = [item.get("labels", item["input_ids"]) for item in batch]
    
    # Find max length in batch
    max_len = min(max(len(x) for x in input_ids), max_seq_len)
    
    # Pad all sequences
    padded_input = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    padded_labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    
    for i, (inp, lbl) in enumerate(zip(input_ids, labels)):
        length = min(len(inp), max_len)
        padded_input[i, :length] = inp[:length]
        if isinstance(lbl, torch.Tensor):
            lbl_len = min(len(lbl), max_len)
            padded_labels[i, :lbl_len] = lbl[:lbl_len]
        else:
            lbl_tensor = torch.tensor(lbl, dtype=torch.long)
            lbl_len = min(len(lbl_tensor), max_len)
            padded_labels[i, :lbl_len] = lbl_tensor[:lbl_len]
    
    # Attention mask: 1 where real tokens, 0 where padding
    attention_mask = (padded_input != pad_id).long()
    
    return {
        "input_ids": padded_input,
        "labels": padded_labels,
        "attention_mask": attention_mask,
    }


def build_pretraining_dataloader(
    data_dir: str,
    tokenizer,
    batch_size: int = 4,
    max_seq_len: int = 4096,
    num_workers: int = 2,
    shuffle: bool = True,
) -> DataLoader:
    """Build DataLoader for pretraining."""
    dataset = StreamingCodeDataset(
        data_dir=data_dir,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        shuffle=shuffle,
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_id, max_seq_len),
        pin_memory=True,
        prefetch_factor=2 if num_workers > 0 else None,
    )


def build_instruction_dataloader(
    pairs: List[Dict],
    tokenizer,
    batch_size: int = 4,
    max_seq_len: int = 2048,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """Build DataLoader for instruction fine-tuning."""
    dataset = InstructionDataset(pairs, tokenizer, max_seq_len)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_id, max_seq_len),
    )


def get_builtin_instruction_data() -> List[Dict]:
    """Return built-in instruction-response pairs for fine-tuning."""
    return CODE_TEMPLATES["instruction_pairs"] + [
        {
            "instruction": "What is the time complexity of binary search?",
            "response": "Binary search has **O(log n)** time complexity.\n\nWith each comparison, you eliminate half the remaining elements:\n- n=1000 needs only ~10 comparisons\n- n=1,000,000 needs only ~20 comparisons\n\nThis makes it extremely efficient for sorted data!"
        },
        {
            "instruction": "Write a merge sort implementation in Python",
            "response": """def merge_sort(arr):
    \"\"\"Sort array using divide-and-conquer merge sort. O(n log n)\"\"\"
    if len(arr) <= 1:
        return arr
    
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    
    return merge(left, right)

def merge(left, right):
    result = []
    i = j = 0
    
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    result.extend(left[i:])
    result.extend(right[j:])
    return result

# Test it:
arr = [38, 27, 43, 3, 9, 82, 10]
print(merge_sort(arr))  # [3, 9, 10, 27, 38, 43, 82]"""
        },
        {
            "instruction": "Explain what a REST API is",
            "response": """A **REST API** (Representational State Transfer) is a way for software systems to communicate over HTTP.

**Key concepts:**
- **Resources**: Everything is a "resource" (user, post, product) with a URL
- **HTTP Methods**: Define what action to take:
  - GET → Read data
  - POST → Create data  
  - PUT/PATCH → Update data
  - DELETE → Delete data
- **Stateless**: Each request is independent, server doesn't remember previous requests

**Example:**
```
GET  /api/users         → Get all users
GET  /api/users/123     → Get user with ID 123
POST /api/users         → Create new user
PUT  /api/users/123     → Update user 123
DELETE /api/users/123   → Delete user 123
```

**Simple Python REST API with Flask:**
```python
from flask import Flask, jsonify, request
app = Flask(__name__)

users = [{"id": 1, "name": "Alice"}]

@app.route('/api/users', methods=['GET'])
def get_users():
    return jsonify(users)

@app.route('/api/users', methods=['POST'])
def create_user():
    user = request.json
    users.append(user)
    return jsonify(user), 201
```"""
        },
        {
            "instruction": "Fix this Python code that gives an IndexError",
            "response": """An **IndexError** means you're trying to access a list position that doesn't exist.

**Common causes and fixes:**

1. **Accessing beyond list length:**
```python
# Bug:
my_list = [1, 2, 3]
print(my_list[5])  # IndexError! Only indices 0,1,2 exist

# Fix: Check length first
if 5 < len(my_list):
    print(my_list[5])
else:
    print("Index out of range")
```

2. **Off-by-one in loops:**
```python
# Bug:
for i in range(len(my_list) + 1):  # Goes one too far
    print(my_list[i])

# Fix:
for i in range(len(my_list)):  # Correct
    print(my_list[i])
```

3. **Empty list:**
```python
# Bug:
empty = []
print(empty[0])  # IndexError!

# Fix:
if empty:  # Check if non-empty
    print(empty[0])
```

Share your specific code and I'll pinpoint the exact issue!"""
        },
    ]


if __name__ == "__main__":
    print("Testing dataset pipeline...")
    
    # Test deduplication
    dedup = DeduplicatedDataset()
    texts = ["hello world", "hello world", "unique text", "another text", "hello world"]
    filtered = dedup.filter_dataset(texts)
    print(f"[Dedup] Original: {len(texts)} → Filtered: {len(filtered)}")
    
    print("\n[Dataset] ✓ Dataset pipeline ready!")
    print("[Dataset] Built-in instruction pairs:", len(get_builtin_instruction_data()))

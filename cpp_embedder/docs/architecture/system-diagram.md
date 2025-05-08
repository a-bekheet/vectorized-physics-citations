# cpp_embedder System Diagram

## Layer Architecture

This document details the component relationships across the four architectural layers.

---

## Complete System View

```
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║   USER APPLICATIONS                                                           ║
║   ─────────────────                                                           ║
║   Python scripts, C++ applications, shell scripts                             ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
                                      │
                                      ▼
╔═══════════════════════════════════════════════════════════════════════════════╗
║                           INTERFACE LAYER                                     ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║                                                                               ║
║  ┌───────────────────┐  ┌───────────────────┐  ┌───────────────────────────┐  ║
║  │   CLI Tool        │  │   C++ API         │  │   Python Bindings         │  ║
║  │   ───────────     │  │   ───────         │  │   ───────────────         │  ║
║  │                   │  │                   │  │                           │  ║
║  │ • Parse args      │  │ • Embedder class  │  │ • embed() function        │  ║
║  │ • Read stdin      │  │ • load_model()    │  │ • load_model()            │  ║
║  │ • Output JSON     │  │ • embed_text()    │  │ • batch_embed()           │  ║
║  │ • Batch mode      │  │ • embed_batch()   │  │ • numpy array output      │  ║
║  │                   │  │ • similarity()    │  │                           │  ║
║  └─────────┬─────────┘  └─────────┬─────────┘  └─────────────┬─────────────┘  ║
║            │                      │                          │                ║
║            └──────────────────────┼──────────────────────────┘                ║
║                                   │                                           ║
╚═══════════════════════════════════╪═══════════════════════════════════════════╝
                                    │
                                    ▼
╔═══════════════════════════════════════════════════════════════════════════════╗
║                             MODEL LAYER                                       ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║                                                                               ║
║  ┌─────────────────────────────────────────────────────────────────────────┐  ║
║  │                         Weight Loader                                   │  ║
║  │                         ─────────────                                   │  ║
║  │  • Read binary weight file                                              │  ║
║  │  • Validate magic number and version                                    │  ║
║  │  • Parse header (dimensions, layer count)                               │  ║
║  │  • Load tensors into memory                                             │  ║
║  │  • Provide weight access by name                                        │  ║
║  └─────────────────────────────────────────────────────────────────────────┘  ║
║                                   │                                           ║
║                                   ▼                                           ║
║  ┌─────────────────────────────────────────────────────────────────────────┐  ║
║  │                      Embedding Tables                                   │  ║
║  │                      ────────────────                                   │  ║
║  │                                                                         │  ║
║  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────┐  │  ║
║  │  │ Word Embeddings │  │ Position Embed. │  │ Token Type Embeddings   │  │  ║
║  │  │ [30522 × 384]   │  │ [512 × 384]     │  │ [2 × 384]               │  │  ║
║  │  └─────────────────┘  └─────────────────┘  └─────────────────────────┘  │  ║
║  │                                                                         │  ║
║  │  Output: word_emb[token_id] + pos_emb[position] + type_emb[type_id]     │  ║
║  └─────────────────────────────────────────────────────────────────────────┘  ║
║                                   │                                           ║
║                                   ▼                                           ║
║  ┌─────────────────────────────────────────────────────────────────────────┐  ║
║  │                    Transformer Encoder Stack                            │  ║
║  │                    ─────────────────────────                            │  ║
║  │                                                                         │  ║
║  │  ┌───────────────────────────────────────────────────────────────────┐  │  ║
║  │  │                    Encoder Layer (×6)                             │  │  ║
║  │  │                                                                   │  │  ║
║  │  │  ┌─────────────────────────────────────────────────────────────┐  │  │  ║
║  │  │  │            Multi-Head Self-Attention                        │  │  │  ║
║  │  │  │                                                             │  │  │  ║
║  │  │  │   Input [seq, 384]                                          │  │  │  ║
║  │  │  │        │                                                    │  │  │  ║
║  │  │  │        ├──→ Q = Linear(384→384) ──┐                         │  │  │  ║
║  │  │  │        ├──→ K = Linear(384→384) ──┼──→ Attention ──→ Output │  │  │  ║
║  │  │  │        └──→ V = Linear(384→384) ──┘        │                │  │  │  ║
║  │  │  │                                            │                │  │  │  ║
║  │  │  │   Attention(Q,K,V) = softmax(QK^T/√32)V   │                │  │  │  ║
║  │  │  │   (12 parallel heads, each 32-dim)         ▼                │  │  │  ║
║  │  │  │                                   Linear(384→384)           │  │  │  ║
║  │  │  └─────────────────────────────────────────────────────────────┘  │  │  ║
║  │  │                              │                                    │  │  ║
║  │  │                              ▼                                    │  │  ║
║  │  │  ┌─────────────────────────────────────────────────────────────┐  │  │  ║
║  │  │  │            Add & LayerNorm                                  │  │  │  ║
║  │  │  │   output = LayerNorm(input + attention_output)              │  │  │  ║
║  │  │  └─────────────────────────────────────────────────────────────┘  │  │  ║
║  │  │                              │                                    │  │  ║
║  │  │                              ▼                                    │  │  ║
║  │  │  ┌─────────────────────────────────────────────────────────────┐  │  │  ║
║  │  │  │            Feed-Forward Network                             │  │  │  ║
║  │  │  │                                                             │  │  │  ║
║  │  │  │   Input [seq, 384]                                          │  │  │  ║
║  │  │  │        │                                                    │  │  │  ║
║  │  │  │        ▼                                                    │  │  │  ║
║  │  │  │   Linear(384 → 1536)                                        │  │  │  ║
║  │  │  │        │                                                    │  │  │  ║
║  │  │  │        ▼                                                    │  │  │  ║
║  │  │  │   GELU activation                                           │  │  │  ║
║  │  │  │        │                                                    │  │  │  ║
║  │  │  │        ▼                                                    │  │  │  ║
║  │  │  │   Linear(1536 → 384)                                        │  │  │  ║
║  │  │  └─────────────────────────────────────────────────────────────┘  │  │  ║
║  │  │                              │                                    │  │  ║
║  │  │                              ▼                                    │  │  ║
║  │  │  ┌─────────────────────────────────────────────────────────────┐  │  │  ║
║  │  │  │            Add & LayerNorm                                  │  │  │  ║
║  │  │  │   output = LayerNorm(input + ff_output)                     │  │  │  ║
║  │  │  └─────────────────────────────────────────────────────────────┘  │  │  ║
║  │  │                                                                   │  │  ║
║  │  └───────────────────────────────────────────────────────────────────┘  │  ║
║  │                                                                         │  ║
║  └─────────────────────────────────────────────────────────────────────────┘  ║
║                                   │                                           ║
║                                   ▼                                           ║
║  ┌─────────────────────────────────────────────────────────────────────────┐  ║
║  │                         Pooling Layer                                   │  ║
║  │                         ─────────────                                   │  ║
║  │                                                                         │  ║
║  │  Input: [seq_len, 384] transformer output + [seq_len] attention mask    │  ║
║  │                                                                         │  ║
║  │  Mean Pooling:                                                          │  ║
║  │    sum = Σ (output[i] * mask[i]) for i in seq_len                       │  ║
║  │    count = Σ mask[i]                                                    │  ║
║  │    pooled = sum / count                                                 │  ║
║  │                                                                         │  ║
║  │  L2 Normalization:                                                      │  ║
║  │    normalized = pooled / ||pooled||₂                                    │  ║
║  │                                                                         │  ║
║  │  Output: [384] normalized embedding vector                              │  ║
║  └─────────────────────────────────────────────────────────────────────────┘  ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
                                    │
                                    ▼
╔═══════════════════════════════════════════════════════════════════════════════╗
║                           TOKENIZER LAYER                                     ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║                                                                               ║
║  ┌─────────────────────────────────────────────────────────────────────────┐  ║
║  │                        Vocabulary Manager                               │  ║
║  │                        ──────────────────                               │  ║
║  │                                                                         │  ║
║  │  ┌─────────────────────────────┐  ┌─────────────────────────────────┐   │  ║
║  │  │ token_to_id                 │  │ id_to_token                     │   │  ║
║  │  │ std::unordered_map          │  │ std::vector<std::string>        │   │  ║
║  │  │ <string, int>               │  │                                 │   │  ║
║  │  │                             │  │                                 │   │  ║
║  │  │ "[PAD]" → 0                 │  │ 0 → "[PAD]"                     │   │  ║
║  │  │ "[UNK]" → 100               │  │ 100 → "[UNK]"                   │   │  ║
║  │  │ "[CLS]" → 101               │  │ 101 → "[CLS]"                   │   │  ║
║  │  │ "[SEP]" → 102               │  │ 102 → "[SEP]"                   │   │  ║
║  │  │ "[MASK]" → 103              │  │ 103 → "[MASK]"                  │   │  ║
║  │  │ "the" → 1996                │  │ ...                             │   │  ║
║  │  │ "##ing" → 2075              │  │                                 │   │  ║
║  │  │ ...                         │  │                                 │   │  ║
║  │  └─────────────────────────────┘  └─────────────────────────────────┘   │  ║
║  │                                                                         │  ║
║  │  Load from: vocab.txt (one token per line, line number = ID)            │  ║
║  └─────────────────────────────────────────────────────────────────────────┘  ║
║                                   │                                           ║
║                                   ▼                                           ║
║  ┌─────────────────────────────────────────────────────────────────────────┐  ║
║  │                      WordPiece Tokenizer                                │  ║
║  │                      ───────────────────                                │  ║
║  │                                                                         │  ║
║  │  Algorithm:                                                             │  ║
║  │  ┌─────────────────────────────────────────────────────────────────┐   │  ║
║  │  │  1. Normalize text (lowercase, strip accents)                   │   │  ║
║  │  │  2. Split on whitespace → words                                 │   │  ║
║  │  │  3. For each word:                                              │   │  ║
║  │  │     a. If word in vocab → emit word                             │   │  ║
║  │  │     b. Else: greedy longest-match from start                    │   │  ║
║  │  │        - Find longest prefix in vocab                           │   │  ║
║  │  │        - Emit token, continue with "##" + remainder             │   │  ║
║  │  │        - If no match found → emit [UNK]                         │   │  ║
║  │  │  4. If word exceeds max_word_length → emit [UNK]                │   │  ║
║  │  └─────────────────────────────────────────────────────────────────┘   │  ║
║  │                                                                         │  ║
║  │  Example: "embedding" → ["em", "##bed", "##ding"]                       │  ║
║  └─────────────────────────────────────────────────────────────────────────┘  ║
║                                   │                                           ║
║                                   ▼                                           ║
║  ┌─────────────────────────────────────────────────────────────────────────┐  ║
║  │                       Token Encoder                                     │  ║
║  │                       ─────────────                                     │  ║
║  │                                                                         │  ║
║  │  Input: List of WordPiece tokens                                        │  ║
║  │                                                                         │  ║
║  │  Processing:                                                            │  ║
║  │  ┌─────────────────────────────────────────────────────────────────┐   │  ║
║  │  │  1. Prepend [CLS] token                                         │   │  ║
║  │  │  2. Append [SEP] token                                          │   │  ║
║  │  │  3. Truncate to max_length (512) if needed                      │   │  ║
║  │  │  4. Convert tokens to IDs via vocabulary lookup                 │   │  ║
║  │  │  5. Generate attention mask (1 for real tokens, 0 for padding)  │   │  ║
║  │  │  6. Pad to max_length with [PAD] tokens if needed               │   │  ║
║  │  │  7. Generate token type IDs (all 0s for single sentence)        │   │  ║
║  │  └─────────────────────────────────────────────────────────────────┘   │  ║
║  │                                                                         │  ║
║  │  Output:                                                                │  ║
║  │    - input_ids: std::vector<int> [max_length]                           │  ║
║  │    - attention_mask: std::vector<int> [max_length]                      │  ║
║  │    - token_type_ids: std::vector<int> [max_length]                      │  ║
║  └─────────────────────────────────────────────────────────────────────────┘  ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
                                    │
                                    ▼
╔═══════════════════════════════════════════════════════════════════════════════╗
║                              MATH LAYER                                       ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║                                                                               ║
║  ┌─────────────────────────────────────────────────────────────────────────┐  ║
║  │                      Matrix Operations                                  │  ║
║  │                      ─────────────────                                  │  ║
║  │                                                                         │  ║
║  │  Data Representation:                                                   │  ║
║  │    All matrices stored as std::vector<float> in row-major order         │  ║
║  │    Shape tracked separately as (rows, cols) pairs                       │  ║
║  │                                                                         │  ║
║  │  Operations:                                                            │  ║
║  │  ┌───────────────────────────────────────────────────────────────────┐  │  ║
║  │  │ matmul(A, B, M, K, N)     Matrix multiply: [M×K] @ [K×N] → [M×N] │  │  ║
║  │  │ transpose(A, rows, cols)  Transpose: [R×C] → [C×R]               │  │  ║
║  │  │ add(A, B, size)           Element-wise addition                   │  │  ║
║  │  │ scale(A, s, size)         Scalar multiplication                   │  │  ║
║  │  │ add_bias(A, bias, ...)    Add bias to each row                    │  │  ║
║  │  └───────────────────────────────────────────────────────────────────┘  │  ║
║  └─────────────────────────────────────────────────────────────────────────┘  ║
║                                                                               ║
║  ┌─────────────────────────────────────────────────────────────────────────┐  ║
║  │                      Activation Functions                               │  ║
║  │                      ────────────────────                               │  ║
║  │                                                                         │  ║
║  │  ┌───────────────────────────────────────────────────────────────────┐  │  ║
║  │  │ gelu(x)                                                           │  │  ║
║  │  │   GELU(x) = x * Φ(x)                                              │  │  ║
║  │  │   Approximation: 0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))          │  │  ║
║  │  │                                                                   │  │  ║
║  │  │ layer_norm(x, gamma, beta, eps=1e-12)                             │  │  ║
║  │  │   μ = mean(x)                                                     │  │  ║
║  │  │   σ² = var(x)                                                     │  │  ║
║  │  │   output = gamma * (x - μ) / √(σ² + ε) + beta                    │  │  ║
║  │  └───────────────────────────────────────────────────────────────────┘  │  ║
║  └─────────────────────────────────────────────────────────────────────────┘  ║
║                                                                               ║
║  ┌─────────────────────────────────────────────────────────────────────────┐  ║
║  │                         Utilities                                       │  ║
║  │                         ─────────                                       │  ║
║  │                                                                         │  ║
║  │  ┌───────────────────────────────────────────────────────────────────┐  │  ║
║  │  │ softmax(x, size)           Softmax over vector                    │  │  ║
║  │  │   max_val = max(x)                                                │  │  ║
║  │  │   exp_x = exp(x - max_val)  (for numerical stability)             │  │  ║
║  │  │   output = exp_x / sum(exp_x)                                     │  │  ║
║  │  │                                                                   │  │  ║
║  │  │ l2_normalize(x, size)      L2 normalization                       │  │  ║
║  │  │   norm = √(Σx²)                                                  │  │  ║
║  │  │   output = x / norm                                               │  │  ║
║  │  │                                                                   │  │  ║
║  │  │ dot(a, b, size)            Dot product                            │  │  ║
║  │  │   output = Σ(a[i] * b[i])                                        │  │  ║
║  │  └───────────────────────────────────────────────────────────────────┘  │  ║
║  └─────────────────────────────────────────────────────────────────────────┘  ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
```

---

## Component Dependencies

```
┌────────────────────────────────────────────────────────────────┐
│                     Dependency Graph                           │
└────────────────────────────────────────────────────────────────┘

                    ┌─────────┐
                    │   CLI   │
                    └────┬────┘
                         │
         ┌───────────────┼───────────────┐
         │               │               │
         ▼               ▼               ▼
    ┌─────────┐    ┌─────────┐    ┌───────────┐
    │ C++ API │    │ Python  │    │           │
    │         │◄───┤ Binding │    │           │
    └────┬────┘    └────┬────┘    │           │
         │              │         │           │
         └──────┬───────┘         │           │
                │                 │           │
                ▼                 │           │
         ┌─────────────┐          │           │
         │   Model     │          │           │
         │   Layer     │          │           │
         └──────┬──────┘          │           │
                │                 │           │
        ┌───────┴───────┐         │           │
        │               │         │           │
        ▼               ▼         │           │
  ┌───────────┐   ┌───────────┐   │           │
  │ Tokenizer │   │  Weight   │   │           │
  │   Layer   │   │  Loader   │   │           │
  └─────┬─────┘   └───────────┘   │           │
        │                         │           │
        └────────────┬────────────┘           │
                     │                        │
                     ▼                        │
              ┌─────────────┐                 │
              │    Math     │◄────────────────┘
              │    Layer    │
              └─────────────┘
                     │
                     ▼
              ┌─────────────┐
              │   C++ STL   │
              │   (only)    │
              └─────────────┘
```

---

## Inter-Component Communication

| From | To | Data Type | Purpose |
|------|----|-----------|---------|
| CLI | C++ API | `std::string` | Input text |
| C++ API | Tokenizer | `std::string` | Text to tokenize |
| Tokenizer | Model | `vector<int>` × 3 | token_ids, attention_mask, token_type_ids |
| Model | Math | `vector<float>` | Tensor data for operations |
| Model | Pooling | `vector<float>` | Sequence embeddings |
| Pooling | C++ API | `vector<float>` | Final embedding (384-dim) |
| C++ API | CLI | `vector<float>` | Embedding for output |
| C++ API | Python | `float*` + size | Raw pointer for numpy conversion |

---

## Thread Safety Model

The current architecture is **single-threaded** by design for simplicity:

- Model weights are read-only after loading (thread-safe for reads)
- All inference computations use local/stack buffers
- No global mutable state
- Future: batch parallelism can be added at the interface layer

---

## Memory Layout

```
┌─────────────────────────────────────────────────────────────┐
│                    Runtime Memory Map                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ Model Weights (loaded once, ~90 MB)                 │    │
│  │                                                     │    │
│  │  • Word embeddings:     30522 × 384 × 4 = 46.9 MB   │    │
│  │  • Position embeddings:   512 × 384 × 4 = 0.8 MB    │    │
│  │  • Encoder layers (×6):                             │    │
│  │    - Attention weights: 4 × 384 × 384 × 4 = 2.4 MB  │    │
│  │    - FFN weights: 384×1536 + 1536×384 = 4.7 MB      │    │
│  │    - Layer norms: negligible                        │    │
│  │  Total per layer: ~7 MB × 6 = 42 MB                 │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ Vocabulary (loaded once, ~3 MB)                     │    │
│  │                                                     │    │
│  │  • Token strings: ~30522 × ~8 chars = ~250 KB       │    │
│  │  • Hash map overhead: ~2.5 MB                       │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ Inference Buffers (per call, ~3 MB peak)            │    │
│  │                                                     │    │
│  │  • Input embeddings:    512 × 384 × 4 = 0.8 MB      │    │
│  │  • Attention scores:    12 × 512 × 512 × 4 = 12 MB  │    │
│  │  • Intermediate:        512 × 1536 × 4 = 3 MB       │    │
│  │  • Output buffer:       384 × 4 = 1.5 KB            │    │
│  │                                                     │    │
│  │  Note: Buffers can be reused across layers          │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                             │
└─────────────────────────────────────────────────────────────┘

Total peak memory: ~110 MB (model + vocab + inference)
```

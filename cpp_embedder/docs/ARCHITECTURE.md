# Architecture

This document provides a technical deep dive into cpp_embedder's architecture, component design, and implementation details.

## Table of Contents

- [Design Principles](#design-principles)
- [Component Overview](#component-overview)
- [Data Flow](#data-flow)
- [Layer Details](#layer-details)
- [Key Design Decisions](#key-design-decisions)
- [Performance Considerations](#performance-considerations)

---

## Design Principles

cpp_embedder follows these core principles:

1. **Zero External Dependencies**: Only C++17 standard library is used
2. **Correctness Over Performance**: Readable, maintainable code prioritized over optimization
3. **Layered Architecture**: Clear separation between math, tokenization, model, and interface layers
4. **Value Semantics**: Tensors use value semantics for safety and simplicity

---

## Component Overview

```
+-----------------------------------------------------------------------------+
|                              INTERFACE LAYER                                 |
+-------------------------+-------------------------+-------------------------+
|     CLI Tool            |   C++ Public API        |      Python Bindings    |
|   (main.cpp)            |   (Embedder class)      |      (ctypes)           |
+------------+------------+------------+------------+------------+------------+
             |                         |                         |
             +-----------+-------------+-----------+-------------+
                         |                         |
                         v                         v
+-----------------------------------------------------------------------------+
|                               MODEL LAYER                                    |
+-----------------------------------------------------------------------------+
|  +-----------------+  +---------------------+  +---------------------------+ |
|  | Weight Loader   |  | Transformer Layers  |  | Pooling & Normalization   | |
|  |                 |  |                     |  |                           | |
|  | - Binary format |  | - Embedding         |  | - Mean pooling            | |
|  | - Validation    |  | - MultiHeadAttention|  | - L2 normalization        | |
|  +-----------------+  | - FeedForward       |  +---------------------------+ |
|                       | - TransformerBlock  |                               |
|                       +---------------------+                               |
+-----------------------------------------------------------------------------+
                                  |
                                  v
+-----------------------------------------------------------------------------+
|                            TOKENIZER LAYER                                   |
+-----------------------------------------------------------------------------+
|  +-----------------+  +---------------------+  +---------------------------+ |
|  | Vocabulary      |  | WordPiece           |  | Token Encoding            | |
|  |                 |  |                     |  |                           | |
|  | - Token->ID map |  | - Subword splitting |  | - [CLS]/[SEP] tokens      | |
|  | - ID->Token map |  | - Unknown handling  |  | - Attention mask          | |
|  +-----------------+  +---------------------+  | - Padding/truncation      | |
|                                               +---------------------------+ |
+-----------------------------------------------------------------------------+
                                  |
                                  v
+-----------------------------------------------------------------------------+
|                              MATH LAYER                                      |
+-----------------------------------------------------------------------------+
|  +-----------------+  +---------------------+  +---------------------------+ |
|  | Tensor Class    |  | Matrix Operations   |  | Activations               | |
|  |                 |  |                     |  |                           | |
|  | - 1D/2D/3D      |  | - matmul            |  | - GELU                    | |
|  | - Row-major     |  | - add, scale        |  | - Softmax                 | |
|  | - Value types   |  | - transpose         |  | - LayerNorm               | |
|  +-----------------+  +---------------------+  +---------------------------+ |
+-----------------------------------------------------------------------------+
```

---

## Data Flow

### Text to Embedding Pipeline

```
Input: "The quick brown fox"
          |
          v
+------------------------------------------------------------------+
| 1. PREPROCESSING                                                  |
|                                                                   |
|    - Convert to lowercase                                         |
|    - Normalize whitespace                                         |
|                                                                   |
|    "the quick brown fox"                                          |
+------------------------------------------------------------------+
          |
          v
+------------------------------------------------------------------+
| 2. TOKENIZATION                                                   |
|                                                                   |
|    Split on whitespace:                                           |
|    ["the", "quick", "brown", "fox"]                               |
|                                                                   |
|    WordPiece encoding:                                            |
|    ["the", "quick", "brown", "fox"]                               |
|    (No subword splits needed for these common words)              |
|                                                                   |
|    Add special tokens:                                            |
|    ["[CLS]", "the", "quick", "brown", "fox", "[SEP]"]             |
|                                                                   |
|    Convert to IDs:                                                |
|    [101, 1996, 4248, 2829, 4419, 102]                             |
|                                                                   |
|    Attention mask:                                                |
|    [1, 1, 1, 1, 1, 1]                                             |
+------------------------------------------------------------------+
          |
          v
+------------------------------------------------------------------+
| 3. EMBEDDING LOOKUP                                               |
|                                                                   |
|    For each token ID:                                             |
|      word_emb = word_embeddings[token_id]      # [384]            |
|      pos_emb  = position_embeddings[position]  # [384]            |
|      type_emb = token_type_embeddings[0]       # [384]            |
|      embedding = word_emb + pos_emb + type_emb                    |
|                                                                   |
|    Apply LayerNorm:                                               |
|      hidden_states = LayerNorm(embeddings)                        |
|                                                                   |
|    Output: [6, 384] tensor (seq_len x hidden_size)                |
+------------------------------------------------------------------+
          |
          v
+------------------------------------------------------------------+
| 4. TRANSFORMER ENCODER (x6 layers)                                |
|                                                                   |
|    For each layer:                                                |
|    +----------------------------------------------------------+   |
|    | Multi-Head Self-Attention (12 heads)                     |   |
|    |                                                          |   |
|    |   Q = hidden @ W_q + b_q    [6, 384]                     |   |
|    |   K = hidden @ W_k + b_k    [6, 384]                     |   |
|    |   V = hidden @ W_v + b_v    [6, 384]                     |   |
|    |                                                          |   |
|    |   Split into 12 heads of dim 32:                         |   |
|    |   Q, K, V: [12, 6, 32]                                    |   |
|    |                                                          |   |
|    |   Attention scores = softmax(Q @ K^T / sqrt(32))         |   |
|    |   Attention output = scores @ V                          |   |
|    |                                                          |   |
|    |   Merge heads: [6, 384]                                  |   |
|    |   Output projection: output @ W_o + b_o                  |   |
|    +----------------------------------------------------------+   |
|                          |                                        |
|                          v                                        |
|    +----------------------------------------------------------+   |
|    | Residual + LayerNorm                                     |   |
|    |   hidden = LayerNorm(hidden + attention_output)          |   |
|    +----------------------------------------------------------+   |
|                          |                                        |
|                          v                                        |
|    +----------------------------------------------------------+   |
|    | Feed-Forward Network                                     |   |
|    |                                                          |   |
|    |   intermediate = GELU(hidden @ W_1 + b_1)   [6, 1536]    |   |
|    |   output = intermediate @ W_2 + b_2         [6, 384]     |   |
|    +----------------------------------------------------------+   |
|                          |                                        |
|                          v                                        |
|    +----------------------------------------------------------+   |
|    | Residual + LayerNorm                                     |   |
|    |   hidden = LayerNorm(hidden + ffn_output)                |   |
|    +----------------------------------------------------------+   |
|                                                                   |
|    Output: [6, 384] tensor                                        |
+------------------------------------------------------------------+
          |
          v
+------------------------------------------------------------------+
| 5. MEAN POOLING                                                   |
|                                                                   |
|    Sum hidden states weighted by attention mask:                  |
|      sum = sum(hidden[i] for i where mask[i] == 1)               |
|      count = sum(mask)                                            |
|      pooled = sum / count                                         |
|                                                                   |
|    Output: [384] vector                                           |
+------------------------------------------------------------------+
          |
          v
+------------------------------------------------------------------+
| 6. L2 NORMALIZATION                                               |
|                                                                   |
|    norm = sqrt(sum(x[i]^2))                                       |
|    normalized = x / norm                                          |
|                                                                   |
|    Output: [384] unit vector                                      |
+------------------------------------------------------------------+
          |
          v
Output: std::vector<float> of size 384
```

---

## Layer Details

### Math Layer (`include/math/`, `src/math/`)

**Tensor Class** (`tensor.hpp`)

```cpp
class Tensor {
    Shape shape_;                    // e.g., {6, 384}
    std::vector<float> data_;        // Row-major storage
};
```

- Supports 1D, 2D, and 3D tensors
- Row-major storage for cache efficiency
- Value semantics with copy/move support
- Bounds-checked access via `at()`

**Operations** (`ops.hpp`)

| Function | Description | Shapes |
|----------|-------------|--------|
| `matmul(A, B)` | Matrix multiplication | (M,K) @ (K,N) -> (M,N) |
| `add(A, B)` | Element-wise addition | Same shapes |
| `scale(A, s)` | Scalar multiplication | Any shape |
| `gelu(x)` | GELU activation | Any shape |
| `softmax(x, axis)` | Softmax normalization | Any shape |
| `layer_norm(x, gamma, beta)` | Layer normalization | Normalizes last dim |
| `transpose(x)` | 2D transpose | (M,N) -> (N,M) |

### Tokenizer Layer (`include/tokenizer/`, `src/tokenizer/`)

**Vocabulary** (`vocab.hpp`)

```cpp
class Vocab {
    std::unordered_map<std::string, TokenId> token_to_id_;
    std::vector<std::string> id_to_token_;
    // Special token IDs: PAD=0, UNK=100, CLS=101, SEP=102
};
```

**WordPiece Tokenizer** (`tokenizer.hpp`)

Algorithm:
1. Lowercase and normalize input
2. Split on whitespace
3. For each word:
   - Try to find longest matching prefix in vocabulary
   - If not found, mark as `[UNK]`
   - For remaining characters, try with `##` prefix
4. Add `[CLS]` at start, `[SEP]` at end
5. Pad or truncate to `max_length`

### Model Layer (`include/model/`, `src/model/`)

**Weight Structures** (`layers.hpp`)

```cpp
struct AttentionWeights {
    Tensor query_weight, query_bias;    // [384, 384], [384]
    Tensor key_weight, key_bias;        // [384, 384], [384]
    Tensor value_weight, value_bias;    // [384, 384], [384]
    Tensor output_weight, output_bias;  // [384, 384], [384]
};

struct TransformerBlockWeights {
    AttentionWeights attention;
    FeedForwardWeights ffn;
    LayerNormWeights attention_layer_norm;
    LayerNormWeights ffn_layer_norm;
};
```

**Multi-Head Attention** (`layers.hpp`)

```cpp
class MultiHeadAttention {
    Tensor forward(const Tensor& input) {
        // input: [seq_len, 384]

        // Project Q, K, V
        auto Q = matmul(input, weights_.query_weight);  // [seq, 384]
        auto K = matmul(input, weights_.key_weight);
        auto V = matmul(input, weights_.value_weight);

        // Split into 12 heads of dim 32
        auto Q_heads = split_heads(Q);  // [12, seq, 32]
        auto K_heads = split_heads(K);
        auto V_heads = split_heads(V);

        // Compute attention for each head
        // scores = softmax(Q @ K^T / sqrt(32))
        // output = scores @ V

        // Merge heads and project
        auto merged = merge_heads(attention_output);  // [seq, 384]
        return matmul(merged, weights_.output_weight);
    }
};
```

**Transformer Block** (`layers.hpp`)

```cpp
class TransformerBlock {
    Tensor forward(const Tensor& input) {
        // Self-attention with residual
        auto attn_out = attention_.forward(input);
        auto hidden = layer_norm(add(input, attn_out), ...);

        // Feed-forward with residual
        auto ffn_out = ffn_.forward(hidden);
        return layer_norm(add(hidden, ffn_out), ...);
    }
};
```

**Embedder** (`embedder.hpp`)

The main class that orchestrates the full pipeline:

```cpp
class Embedder {
    std::vector<float> embed(const std::string& text) {
        // 1. Tokenize
        auto tokens = tokenizer_->encode(text, config_.max_seq_length);

        // 2. Embedding lookup
        auto hidden = embedding_layer_->forward(tokens);

        // 3. Pass through transformer layers
        for (auto& block : transformer_blocks_) {
            hidden = block->forward(hidden);
        }

        // 4. Mean pooling
        auto pooled = mean_pool(hidden, attention_mask);

        // 5. Normalize
        if (config_.normalize_embeddings) {
            l2_normalize(pooled);
        }

        return pooled;
    }
};
```

### Interface Layer

**C API** (`bindings/c_api.h`)

Provides C-compatible interface for foreign language bindings:

```c
// Lifecycle
EmbedderHandle embedder_create(const char* weights_path);
void embedder_destroy(EmbedderHandle handle);

// Embedding
CppEmbedderError embedder_embed(
    EmbedderHandle handle,
    const char* text,
    float* output,
    uint32_t output_size
);
```

**Python Bindings** (`python/cpp_embedder/embedder.py`)

Uses ctypes to wrap the C API:

```python
class Embedder:
    def __init__(self, weights_path):
        self._handle = _lib.embedder_create(weights_path.encode())

    def embed(self, text):
        output = np.empty(384, dtype=np.float32)
        _lib.embedder_embed(self._handle, text.encode(), output.ctypes, 384)
        return output
```

---

## Key Design Decisions

### ADR-001: C++17 Standard

**Rationale**: C++17 provides `std::string_view`, `std::optional`, and `std::filesystem` without requiring external dependencies.

### ADR-002: Custom Binary Weight Format

**Rationale**: Avoids dependencies on HDF5, NumPy, or SafeTensors. Simple format enables direct memory mapping.

Format structure:
```
[Header: magic, version, tensor_count]
[Tensor entries: name, shape, offset, size]
[Tensor data: raw float32 values]
[Optional: vocabulary tokens]
```

### ADR-003: Row-Major Tensor Layout

**Rationale**: Matches C++ array conventions and enables cache-efficient row-wise iteration.

### ADR-004: Value Semantics for Tensors

**Rationale**: Simplifies memory management and prevents aliasing bugs. Performance impact is acceptable for the target use case.

### ADR-005: No SIMD/Threading (Initial Version)

**Rationale**: Prioritizes correctness, readability, and portability. Optimization can be added incrementally.

---

## Performance Considerations

### Memory Layout

Tensors use row-major storage for cache-friendly sequential access:

```
Matrix A[M][N]:
Memory: [A[0][0], A[0][1], ..., A[0][N-1], A[1][0], ...]

Access pattern for matrix multiply:
for i in M:
    for j in N:
        for k in K:
            C[i][j] += A[i][k] * B[k][j]  // B access is strided
```

### Bottlenecks

1. **Matrix multiplication**: O(n^3) operations dominate inference time
2. **Attention**: O(seq_len^2) memory for attention scores
3. **Weight loading**: One-time cost at startup (~90MB file)

### Typical Timings (unoptimized)

| Operation | Time |
|-----------|------|
| Tokenization | ~1ms |
| Embedding lookup | ~5ms |
| Single transformer layer | ~50-100ms |
| Full inference (6 layers) | ~300-600ms |
| Mean pooling + normalize | ~1ms |

### Optimization Opportunities

1. **SIMD**: Use AVX/SSE for matrix operations (10-20x speedup possible)
2. **Threading**: Parallelize batch processing
3. **Memory reuse**: Pre-allocate intermediate tensors
4. **Quantization**: Use INT8 weights (requires quantized model)
5. **GPU**: CUDA/Metal backends for large batches

---

## Model Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `HIDDEN_SIZE` | 384 | Embedding dimension |
| `INTERMEDIATE_SIZE` | 1536 | FFN hidden dimension |
| `NUM_ATTENTION_HEADS` | 12 | Attention head count |
| `HEAD_DIM` | 32 | Dimension per head |
| `NUM_LAYERS` | 6 | Transformer layers |
| `MAX_POSITION_EMBEDDINGS` | 512 | Maximum sequence length |
| `VOCAB_SIZE` | 30522 | Vocabulary size |

---

## File Dependencies

```
embedder.hpp
├── weights.hpp
│   └── ../math/tensor.hpp
├── layers.hpp
│   ├── ../math/tensor.hpp
│   └── ../math/ops.hpp
└── ../tokenizer/tokenizer.hpp
    └── vocab.hpp
```

---

## Testing Strategy

Unit tests cover each layer independently:

- `test_math.cpp`: Tensor operations, activations
- `test_tokenizer.cpp`: WordPiece algorithm, encoding/decoding
- `test_layers.cpp`: Attention, FFN, transformer blocks

Integration tests verify end-to-end correctness:

- Compare embeddings with reference PyTorch implementation
- Validate similarity scores for known sentence pairs

# cpp_embedder Architecture Overview

## Purpose

cpp_embedder is a pure C++17 implementation of a sentence embedding system compatible with the all-MiniLM-L6-v2 model architecture. It transforms arbitrary text into fixed-dimensional dense vectors (384 dimensions) suitable for semantic similarity tasks.

## Design Principles

1. **Zero External Dependencies**: Uses only C++17 standard library
2. **Simplicity Over Performance**: Readable, maintainable code prioritized over optimization
3. **Layered Architecture**: Clear separation between math, tokenization, model, and interface layers
4. **Single-Header Friendly**: Components designed for easy integration

---

## Component Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              INTERFACE LAYER                                │
├─────────────────────┬─────────────────────┬─────────────────────────────────┤
│     CLI Tool        │   C++ Public API    │      Python Bindings            │
│   (standalone)      │   (Embedder class)  │      (pybind11/ctypes)          │
└─────────┬───────────┴─────────┬───────────┴─────────────┬───────────────────┘
          │                     │                         │
          └─────────────────────┼─────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                               MODEL LAYER                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────────┐  │
│  │ Weight Loader   │  │ Transformer     │  │ Pooling                     │  │
│  │                 │  │                 │  │                             │  │
│  │ - Binary format │  │ - Encoder stack │  │ - Mean pooling              │  │
│  │ - Memory map    │  │ - Attention     │  │ - Normalization             │  │
│  │ - Validation    │  │ - Feed-forward  │  │                             │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────────────────┘  │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            TOKENIZER LAYER                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────────┐  │
│  │ Vocabulary      │  │ WordPiece       │  │ Token Encoding              │  │
│  │                 │  │                 │  │                             │  │
│  │ - Token->ID map │  │ - Subword split │  │ - Special tokens [CLS]/[SEP]│  │
│  │ - ID->Token map │  │ - Unknown token │  │ - Attention mask            │  │
│  │ - Vocab loader  │  │ - Max word len  │  │ - Padding/truncation        │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────────────────┘  │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              MATH LAYER                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────────┐  │
│  │ Matrix/Vector   │  │ Activations     │  │ Utilities                   │  │
│  │                 │  │                 │  │                             │  │
│  │ - MatMul        │  │ - GELU          │  │ - Softmax                   │  │
│  │ - Transpose     │  │ - LayerNorm     │  │ - L2 Normalize              │  │
│  │ - Add/Scale     │  │                 │  │ - Batch operations          │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow

### Text to Embedding Pipeline

```
┌──────────────┐
│ Input Text   │  "The quick brown fox"
└──────┬───────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│ 1. TOKENIZATION                                              │
│                                                              │
│    Text → Lowercase → Split → WordPiece → Add Special Tokens │
│                                                              │
│    "The quick brown fox"                                     │
│         ↓                                                    │
│    ["the", "quick", "brown", "fox"]                          │
│         ↓                                                    │
│    ["[CLS]", "the", "quick", "brown", "fox", "[SEP]"]        │
│         ↓                                                    │
│    [101, 1996, 4248, 2829, 4419, 102]  (token IDs)           │
│    [1, 1, 1, 1, 1, 1]                   (attention mask)     │
└──────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│ 2. EMBEDDING LOOKUP                                          │
│                                                              │
│    Token IDs → Word Embeddings (384-dim each)                │
│             + Position Embeddings                            │
│             + Token Type Embeddings                          │
│                                                              │
│    Output: [seq_len, 384] matrix                             │
└──────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│ 3. TRANSFORMER ENCODER (×6 layers)                           │
│                                                              │
│    For each layer:                                           │
│    ┌─────────────────────────────────────────────────────┐   │
│    │ Multi-Head Self-Attention (12 heads)                │   │
│    │    Q, K, V projections → Attention → Output proj    │   │
│    └─────────────────────────────────────────────────────┘   │
│                          ↓                                   │
│    ┌─────────────────────────────────────────────────────┐   │
│    │ Add & LayerNorm                                     │   │
│    └─────────────────────────────────────────────────────┘   │
│                          ↓                                   │
│    ┌─────────────────────────────────────────────────────┐   │
│    │ Feed-Forward Network (384 → 1536 → 384)             │   │
│    │    Linear → GELU → Linear                           │   │
│    └─────────────────────────────────────────────────────┘   │
│                          ↓                                   │
│    ┌─────────────────────────────────────────────────────┐   │
│    │ Add & LayerNorm                                     │   │
│    └─────────────────────────────────────────────────────┘   │
│                                                              │
│    Output: [seq_len, 384] matrix                             │
└──────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│ 4. POOLING                                                   │
│                                                              │
│    Mean pooling over sequence (with attention mask)          │
│    [seq_len, 384] → [384]                                    │
│                                                              │
│    L2 Normalization                                          │
│    [384] → [384] (unit vector)                               │
└──────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────┐
│ Output       │  std::vector<float> of size 384
│ Embedding    │  (normalized unit vector)
└──────────────┘
```

---

## Module Boundaries

### Math Layer
- **Responsibility**: All numerical computations
- **Dependencies**: None (C++ STL only)
- **Exports**: Matrix operations, activation functions, normalization
- **Data Types**: `std::vector<float>` for all tensors (flattened)

### Tokenizer Layer
- **Responsibility**: Text to token ID conversion
- **Dependencies**: Math Layer (none currently)
- **Exports**: Tokenizer class with `encode()` method
- **Data Types**: `std::vector<int>` for token IDs, `std::string` for text

### Model Layer
- **Responsibility**: Neural network inference
- **Dependencies**: Math Layer, Tokenizer Layer
- **Exports**: Model class with `forward()` method
- **Data Types**: Uses Math Layer types for all tensors

### Interface Layer
- **Responsibility**: User-facing APIs
- **Dependencies**: Model Layer, Tokenizer Layer
- **Exports**:
  - `Embedder` class for C++ users
  - Python module for Python users
  - CLI executable

---

## Model Specifications (all-MiniLM-L6-v2)

| Parameter | Value |
|-----------|-------|
| Hidden Size | 384 |
| Intermediate Size | 1536 |
| Number of Attention Heads | 12 |
| Number of Hidden Layers | 6 |
| Max Position Embeddings | 512 |
| Vocabulary Size | 30522 |
| Output Embedding Size | 384 |

---

## File Organization

```
cpp_embedder/
├── docs/
│   ├── architecture/      # This document lives here
│   ├── interfaces/        # API specifications
│   └── decisions/         # ADRs and tech choices
├── src/
│   ├── math/              # Matrix ops, activations
│   ├── tokenizer/         # WordPiece implementation
│   ├── model/             # Transformer components
│   ├── api/               # Public C++ API
│   └── cli/               # Command-line tool
├── include/               # Public headers
├── bindings/              # Python bindings
├── weights/               # Model weight files (binary)
├── vocab/                 # Vocabulary files
└── tests/                 # Unit and integration tests
```

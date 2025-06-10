# Usage Guide

This guide covers using cpp_embedder via the CLI, C++ API, and Python bindings.

## Table of Contents

- [Converting Weights](#converting-weights)
- [Command-Line Interface](#command-line-interface)
- [C++ API](#c-api)
- [Python Bindings](#python-bindings)
- [Common Tasks](#common-tasks)

---

## Converting Weights

Before using cpp_embedder, you need to convert PyTorch model weights to the binary format.

### Using the Conversion Script

```bash
python tools/convert_weights.py \
    --model sentence-transformers/all-MiniLM-L6-v2 \
    --output model.weights
```

### Options

| Option | Description |
|--------|-------------|
| `--model` | HuggingFace model name or local path |
| `--output` | Output file path (default: `model.weights`) |
| `--vocab` | Optional: separate vocab file output |

### Manual Conversion

If the script is not available, the weight file format is documented in `docs/interfaces/weight-format.md`. The format is:

1. Header: magic number "EMBD", version, tensor count
2. Tensor index: name, shape, data offset for each tensor
3. Tensor data: raw float32 values in row-major order
4. Optional: vocabulary tokens

---

## Command-Line Interface

The `cpp_embed` CLI provides a simple way to generate embeddings.

### Basic Usage

```bash
# Embed a single text
./cpp_embed -m model.weights -t "Hello, world!"

# Output (JSON):
{
  "embeddings": [
    {
      "text": "Hello, world!",
      "vector": [0.0123, -0.0456, ...]
    }
  ]
}
```

### Options

| Option | Description |
|--------|-------------|
| `-m, --model PATH` | Path to weights file (required) |
| `-t, --text TEXT` | Text to embed (can be repeated) |
| `-f, --file PATH` | File with texts (one per line) |
| `-o, --output PATH` | Output file (stdout if omitted) |
| `--format FORMAT` | Output format: `json` (default) or `binary` |
| `--similarity T1 T2` | Compute similarity between two texts |
| `-h, --help` | Show help message |

### Examples

**Multiple texts:**

```bash
./cpp_embed -m model.weights -t "First sentence" -t "Second sentence"
```

**From file:**

```bash
# texts.txt contains one sentence per line
./cpp_embed -m model.weights -f texts.txt -o embeddings.json
```

**Compute similarity:**

```bash
./cpp_embed -m model.weights --similarity "hello" "hi there"

# Output:
Text 1: "hello"
Text 2: "hi there"
Cosine similarity: 0.654321
```

**Binary output:**

```bash
./cpp_embed -m model.weights -f texts.txt --format binary -o embeddings.bin
```

Binary format:
- 4 bytes: number of embeddings (uint32)
- 4 bytes: embedding dimension (uint32)
- N * 384 * 4 bytes: raw float32 vectors

---

## C++ API

### Headers

```cpp
#include "model/embedder.hpp"    // Main embedder class
#include "tokenizer/tokenizer.hpp"  // Tokenizer access
#include "math/tensor.hpp"       // Tensor operations (optional)
```

### Creating an Embedder

```cpp
using namespace cpp_embedder::model;

// Option 1: Using config struct
EmbedderConfig config;
config.weights_path = "model.weights";
config.vocab_path = "vocab.txt";  // Optional if embedded in weights
config.max_seq_length = 256;
config.normalize_embeddings = true;

Embedder embedder(config);

// Option 2: Direct paths
Embedder embedder("model.weights", "vocab.txt");

// Check if loaded
if (!embedder.is_loaded()) {
    std::cerr << "Failed to load model\n";
    return 1;
}
```

### Generating Embeddings

```cpp
// Single text
std::vector<float> embedding = embedder.embed("Hello, world!");
// embedding.size() == 384

// Batch of texts
std::vector<std::string> texts = {
    "First sentence",
    "Second sentence",
    "Third sentence"
};
std::vector<std::vector<float>> embeddings = embedder.embed_batch(texts);
// embeddings.size() == 3, each inner vector has size 384
```

### Computing Similarity

```cpp
// Cosine similarity between two embeddings
float similarity = cosine_similarity(embeddings[0], embeddings[1]);
// Returns value in [-1, 1]

// Dot product (for normalized embeddings, same as cosine similarity)
float dot = dot_product(embeddings[0], embeddings[1]);

// Normalize an embedding (if not already normalized)
std::vector<float> vec = {...};
normalize_embedding(vec);  // In-place normalization
```

### Accessing the Tokenizer

```cpp
const auto& tokenizer = embedder.tokenizer();

// Tokenize text
std::vector<std::string> tokens = tokenizer.tokenize("Hello, world!");
// ["[CLS]", "hello", ",", "world", "!", "[SEP]"]

// Encode to token IDs
std::vector<int> ids = tokenizer.encode("Hello, world!", 256);
// [101, 7592, 1010, 2088, 999, 102, 0, 0, ...] (padded to max_length)

// Decode back to text
std::string text = tokenizer.decode(ids);
```

### Error Handling

```cpp
try {
    Embedder embedder(config);
    auto embedding = embedder.embed("test");
} catch (const std::runtime_error& e) {
    std::cerr << "Error: " << e.what() << std::endl;
}
```

### Complete Example

```cpp
#include "model/embedder.hpp"
#include <iostream>
#include <vector>

int main() {
    using namespace cpp_embedder::model;

    // Load model
    EmbedderConfig config;
    config.weights_path = "model.weights";
    Embedder embedder(config);

    if (!embedder.is_loaded()) {
        std::cerr << "Failed to load model\n";
        return 1;
    }

    // Embed some sentences
    std::vector<std::string> sentences = {
        "The cat sits on the mat.",
        "A feline rests on the rug.",
        "The stock market crashed today."
    };

    auto embeddings = embedder.embed_batch(sentences);

    // Find most similar pair
    float max_sim = -1;
    int best_i = 0, best_j = 0;

    for (int i = 0; i < sentences.size(); ++i) {
        for (int j = i + 1; j < sentences.size(); ++j) {
            float sim = cosine_similarity(embeddings[i], embeddings[j]);
            if (sim > max_sim) {
                max_sim = sim;
                best_i = i;
                best_j = j;
            }
        }
    }

    std::cout << "Most similar sentences:\n";
    std::cout << "  \"" << sentences[best_i] << "\"\n";
    std::cout << "  \"" << sentences[best_j] << "\"\n";
    std::cout << "Similarity: " << max_sim << "\n";

    return 0;
}
```

---

## Python Bindings

### Installation

```bash
cd python
pip install .
```

### Basic Usage

```python
from cpp_embedder import Embedder

# Load model
embedder = Embedder("model.weights")

# Check model is loaded
print(embedder.is_loaded)  # True
print(embedder.embedding_dim)  # 384
print(embedder.max_seq_length)  # 256
```

### Generating Embeddings

```python
import numpy as np

# Single text - returns numpy array of shape (384,)
vec = embedder.embed("Hello, world!")
print(vec.shape)  # (384,)
print(vec.dtype)  # float32

# Multiple texts - returns array of shape (n, 384)
vecs = embedder.embed_batch(["Hello", "World", "Test"])
print(vecs.shape)  # (3, 384)

# Using the unified embed() method
vec = embedder.embed("Single text")  # shape (384,)
vecs = embedder.embed(["Multiple", "texts"])  # shape (2, 384)
```

### Computing Similarity

```python
from cpp_embedder import cosine_similarity, find_similar, normalize

# Similarity between two vectors
sim = cosine_similarity(vecs[0], vecs[1])  # float

# Similarity between one query and many corpus vectors
sims = cosine_similarity(vecs[0], vecs)  # shape (3,)

# Find top-k most similar
indices, scores = find_similar(query=vecs[0], corpus=vecs, top_k=2)

# Normalize embeddings
normalized = normalize(vecs)  # shape (n, 384), unit vectors
```

### Accessing the Tokenizer

```python
tokenizer = embedder.tokenizer

# Encode text to token IDs
token_ids = tokenizer.encode("Hello, world!", max_length=256)
print(token_ids)  # [101, 7592, 1010, 2088, 999, 102]

# Decode back to text
text = tokenizer.decode(token_ids)
print(text)  # "hello , world !"

# Special token IDs
print(tokenizer.cls_token_id)  # 101
print(tokenizer.sep_token_id)  # 102
print(tokenizer.pad_token_id)  # 0
print(tokenizer.unk_token_id)  # 100
```

### Model Metadata

```python
# Get model information
metadata = embedder.metadata
print(metadata)
# {
#     'model_name': 'all-MiniLM-L6-v2',
#     'model_version': '1.0.0',
#     'embedding_dim': 384,
#     'vocab_size': 30522,
#     'num_layers': 6,
#     'num_attention_heads': 12,
#     'max_seq_length': 256
# }
```

### Configuration Options

```python
embedder = Embedder(
    weights_path="model.weights",
    max_seq_length=128,       # Shorter sequences = faster
    normalize_embeddings=True, # L2 normalize output
    num_threads=4             # Number of threads (0=auto)
)
```

### Context Manager

```python
with Embedder("model.weights") as embedder:
    vec = embedder.embed("Hello")
# Resources automatically released
```

### Complete Example

```python
from cpp_embedder import Embedder, cosine_similarity, find_similar
import numpy as np

def semantic_search(query: str, documents: list[str], embedder: Embedder, top_k: int = 5):
    """Find most similar documents to a query."""
    # Embed query and documents
    query_vec = embedder.embed(query)
    doc_vecs = embedder.embed_batch(documents)

    # Find top-k similar
    indices, scores = find_similar(query_vec, doc_vecs, top_k)

    results = []
    for idx, score in zip(indices, scores):
        results.append({
            'document': documents[idx],
            'score': float(score)
        })
    return results


if __name__ == "__main__":
    # Load model
    embedder = Embedder("model.weights")

    # Sample documents
    documents = [
        "Machine learning is a subset of artificial intelligence.",
        "Python is a popular programming language.",
        "Deep learning uses neural networks with many layers.",
        "The weather is nice today.",
        "Natural language processing deals with text data."
    ]

    # Search
    query = "What is AI?"
    results = semantic_search(query, documents, embedder, top_k=3)

    print(f"Query: {query}\n")
    print("Top results:")
    for i, r in enumerate(results, 1):
        print(f"  {i}. [{r['score']:.4f}] {r['document']}")
```

---

## Common Tasks

### Semantic Similarity

Compare how similar two pieces of text are:

```python
embedder = Embedder("model.weights")

text1 = "The cat sat on the mat."
text2 = "A feline rested on the rug."
text3 = "The stock market crashed."

vec1 = embedder.embed(text1)
vec2 = embedder.embed(text2)
vec3 = embedder.embed(text3)

print(f"Similarity 1-2: {cosine_similarity(vec1, vec2):.4f}")  # High
print(f"Similarity 1-3: {cosine_similarity(vec1, vec3):.4f}")  # Low
```

### Document Clustering

Group similar documents together:

```python
from sklearn.cluster import KMeans

documents = ["...", "...", ...]  # Your documents
embeddings = embedder.embed_batch(documents)

kmeans = KMeans(n_clusters=5)
clusters = kmeans.fit_predict(embeddings)
```

### Duplicate Detection

Find near-duplicate texts:

```python
def find_duplicates(texts, threshold=0.95):
    embeddings = embedder.embed_batch(texts)
    duplicates = []

    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            sim = cosine_similarity(embeddings[i], embeddings[j])
            if sim > threshold:
                duplicates.append((i, j, sim))

    return duplicates
```

### Building a Search Index

For large-scale similarity search, use a vector database or approximate nearest neighbor library:

```python
import faiss

# Generate embeddings
embeddings = embedder.embed_batch(documents)
embeddings = np.asarray(embeddings, dtype='float32')

# Build FAISS index
index = faiss.IndexFlatIP(384)  # Inner product (same as cosine for normalized)
index.add(embeddings)

# Search
query = embedder.embed("search query")
query = np.asarray([query], dtype='float32')
distances, indices = index.search(query, k=10)
```

### Batch Processing Large Files

```python
def process_large_file(input_path, output_path, batch_size=100):
    embedder = Embedder("model.weights")

    with open(input_path) as f:
        lines = [line.strip() for line in f if line.strip()]

    all_embeddings = []
    for i in range(0, len(lines), batch_size):
        batch = lines[i:i+batch_size]
        embeddings = embedder.embed_batch(batch)
        all_embeddings.append(embeddings)

    all_embeddings = np.vstack(all_embeddings)
    np.save(output_path, all_embeddings)
```

---

## Limitations

1. **Sequence length**: Maximum 512 tokens. Longer texts are truncated.
2. **Performance**: Pure C++ without SIMD optimization. For production workloads, consider GPU-accelerated alternatives.
3. **Model**: Only compatible with all-MiniLM-L6-v2 architecture.
4. **Language**: Optimized for English text (following the original model's training).

# cpp_embedder

A pure C++17 implementation of sentence embeddings using the all-MiniLM-L6-v2 architecture. Zero external dependencies - just standard C++ library.

## Overview

cpp_embedder transforms text into 384-dimensional dense vectors suitable for semantic similarity tasks. It implements the full pipeline: tokenization, transformer encoding, mean pooling, and normalization.

**Key features:**

- Pure C++17, no external dependencies
- Compatible with all-MiniLM-L6-v2 model weights
- WordPiece tokenizer implementation
- 384-dimensional normalized embeddings
- Command-line interface
- Python bindings (via ctypes)
- Cross-platform (Linux, macOS, Windows)

## Quick Start

### Building

```bash
# Clone and build
cd cpp_embedder
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j

# Run tests
./cpp_embedder_tests
```

### Converting Weights

Before using the embedder, convert PyTorch weights to the binary format:

```bash
python tools/convert_weights.py \
    --model sentence-transformers/all-MiniLM-L6-v2 \
    --output model.weights
```

### Using the CLI

```bash
# Embed a single text
./cpp_embed -m model.weights -t "Hello, world!"

# Embed multiple texts
./cpp_embed -m model.weights -t "First sentence" -t "Second sentence"

# Compute similarity
./cpp_embed -m model.weights --similarity "hello" "hi there"

# Process a file of texts
./cpp_embed -m model.weights -f texts.txt -o embeddings.json
```

### Using the C++ API

```cpp
#include "model/embedder.hpp"

using namespace cpp_embedder::model;

int main() {
    // Load model
    EmbedderConfig config;
    config.weights_path = "model.weights";
    Embedder embedder(config);

    // Generate embedding
    auto embedding = embedder.embed("This is a test sentence.");
    // embedding is std::vector<float> of size 384

    // Batch embedding
    std::vector<std::string> texts = {"Hello", "World"};
    auto embeddings = embedder.embed_batch(texts);

    // Compute similarity
    float sim = cosine_similarity(embeddings[0], embeddings[1]);

    return 0;
}
```

### Using Python

```python
from cpp_embedder import Embedder, cosine_similarity

# Load model
embedder = Embedder("model.weights")

# Single embedding
vec = embedder.embed("Hello, world!")  # numpy array, shape (384,)

# Batch embedding
vecs = embedder.embed_batch(["Hello", "World"])  # shape (2, 384)

# Find similarity
sim = cosine_similarity(vecs[0], vecs[1])
```

## Model Specifications

| Parameter | Value |
|-----------|-------|
| Architecture | BERT-based transformer |
| Hidden Size | 384 |
| Intermediate Size | 1536 |
| Attention Heads | 12 |
| Encoder Layers | 6 |
| Vocabulary Size | 30,522 |
| Max Sequence Length | 512 |
| Output Dimension | 384 |

## Project Structure

```
cpp_embedder/
├── include/                 # Public headers
│   ├── math/               # Tensor and math operations
│   ├── tokenizer/          # WordPiece tokenizer
│   ├── model/              # Transformer layers and embedder
│   ├── bindings/           # C API for foreign language bindings
│   └── cli/                # Command-line argument parsing
├── src/                    # Implementation files
├── python/                 # Python package
├── tests/                  # Unit tests
├── tools/                  # Weight conversion scripts
└── docs/                   # Documentation
    ├── BUILD.md           # Build instructions
    ├── USAGE.md           # Usage guide
    └── ARCHITECTURE.md    # Technical deep dive
```

## Documentation

- [Build Instructions](docs/BUILD.md) - Prerequisites and build options
- [Usage Guide](docs/USAGE.md) - CLI, C++ API, and Python bindings
- [Architecture](docs/ARCHITECTURE.md) - Technical design and data flow

## Requirements

- C++17 compatible compiler (GCC 7+, Clang 5+, MSVC 19.14+)
- CMake 3.16+
- Python 3.8+ with NumPy (for Python bindings)

## Performance

This implementation prioritizes correctness and readability over raw performance. It uses pure scalar C++ without SIMD or threading. Typical inference time:

- Single sentence: 100-500ms
- Suitable for batch processing, prototyping, and educational use

## License

MIT License

## Acknowledgments

- [sentence-transformers](https://github.com/UKPLab/sentence-transformers) for the original all-MiniLM-L6-v2 model
- [Hugging Face](https://huggingface.co/) for model hosting and tokenizer specifications

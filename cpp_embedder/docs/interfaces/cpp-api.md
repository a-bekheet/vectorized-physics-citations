# C++ API Interface Contract

## Overview

This document defines the public C++ API for the `cpp_embedder` library, a pure C++17 implementation of sentence embeddings compatible with `all-MiniLM-L6-v2`.

## Namespace

All public APIs reside in the `embedder` namespace.

```cpp
namespace embedder {
    // All public types and classes defined here
}
```

---

## Configuration Structures

### `EmbedderConfig`

Configuration for the main embedder.

```cpp
namespace embedder {

struct EmbedderConfig {
    // Path to the binary weight file
    std::string weights_path;

    // Maximum sequence length (input will be truncated)
    // Default: 256 (model's trained max), Range: [1, 512]
    uint32_t max_seq_length = 256;

    // Whether to normalize output embeddings to unit length
    // Default: true (cosine similarity ready)
    bool normalize_embeddings = true;

    // Number of threads for parallel computation
    // Default: 0 (auto-detect based on hardware)
    uint32_t num_threads = 0;
};

} // namespace embedder
```

### `TokenizerConfig`

Configuration for the WordPiece tokenizer.

```cpp
namespace embedder {

struct TokenizerConfig {
    // Path to vocabulary file (text format, one token per line)
    // If empty, vocabulary is loaded from weight file
    std::string vocab_path;

    // Whether to lowercase input text before tokenization
    // Default: true (matches uncased model)
    bool do_lower_case = true;

    // Maximum word length before splitting into characters
    // Default: 200
    uint32_t max_chars_per_word = 200;

    // Unknown token string
    // Default: "[UNK]"
    std::string unk_token = "[UNK]";

    // Separator token string
    // Default: "[SEP]"
    std::string sep_token = "[SEP]";

    // Classification token string
    // Default: "[CLS]"
    std::string cls_token = "[CLS]";

    // Padding token string
    // Default: "[PAD]"
    std::string pad_token = "[PAD]";
};

} // namespace embedder
```

---

## Error Handling

### `ErrorCode`

Enumeration of possible error conditions.

```cpp
namespace embedder {

enum class ErrorCode {
    // Success
    OK = 0,

    // File I/O errors
    FILE_NOT_FOUND = 1,
    FILE_READ_ERROR = 2,
    FILE_FORMAT_ERROR = 3,

    // Weight file errors
    INVALID_MAGIC_NUMBER = 10,
    UNSUPPORTED_VERSION = 11,
    WEIGHT_MISMATCH = 12,
    CHECKSUM_MISMATCH = 13,

    // Tokenizer errors
    VOCAB_NOT_LOADED = 20,
    TOKEN_NOT_FOUND = 21,
    SEQUENCE_TOO_LONG = 22,

    // Inference errors
    MODEL_NOT_LOADED = 30,
    INVALID_INPUT = 31,
    COMPUTATION_ERROR = 32,

    // Memory errors
    ALLOCATION_FAILED = 40,

    // Configuration errors
    INVALID_CONFIG = 50
};

} // namespace embedder
```

### `Result<T>`

Result type for operations that may fail.

```cpp
namespace embedder {

template<typename T>
class Result {
public:
    // Check if operation succeeded
    bool ok() const noexcept;

    // Check if operation failed
    bool is_error() const noexcept;

    // Get the value (undefined behavior if is_error())
    const T& value() const&;
    T& value() &;
    T&& value() &&;

    // Get the error code (undefined behavior if ok())
    ErrorCode error() const noexcept;

    // Get human-readable error message
    std::string error_message() const;

    // Static factory methods
    static Result<T> success(T value);
    static Result<T> failure(ErrorCode code, std::string message = "");
};

// Specialization for void return type
template<>
class Result<void> {
public:
    bool ok() const noexcept;
    bool is_error() const noexcept;
    ErrorCode error() const noexcept;
    std::string error_message() const;

    static Result<void> success();
    static Result<void> failure(ErrorCode code, std::string message = "");
};

} // namespace embedder
```

---

## Tokenizer Interface

### `TokenizedInput`

Structure representing tokenized text ready for model inference.

```cpp
namespace embedder {

struct TokenizedInput {
    // Token IDs (vocabulary indices)
    std::vector<uint32_t> input_ids;

    // Attention mask (1 for real tokens, 0 for padding)
    std::vector<uint32_t> attention_mask;

    // Token type IDs (0 for first sequence, 1 for second)
    std::vector<uint32_t> token_type_ids;

    // Original tokens before conversion to IDs (for debugging)
    std::vector<std::string> tokens;

    // Sequence length (excluding padding)
    size_t length() const noexcept;

    // Padded sequence length
    size_t padded_length() const noexcept;
};

} // namespace embedder
```

### `Tokenizer`

WordPiece tokenizer class.

```cpp
namespace embedder {

class Tokenizer {
public:
    // Default constructor (vocabulary must be loaded separately)
    Tokenizer();

    // Constructor with configuration
    explicit Tokenizer(const TokenizerConfig& config);

    // Non-copyable
    Tokenizer(const Tokenizer&) = delete;
    Tokenizer& operator=(const Tokenizer&) = delete;

    // Movable
    Tokenizer(Tokenizer&&) noexcept;
    Tokenizer& operator=(Tokenizer&&) noexcept;

    ~Tokenizer();

    // Load vocabulary from text file (one token per line)
    Result<void> load_vocab(const std::string& path);

    // Load vocabulary from raw data (embedded in weight file)
    Result<void> load_vocab_from_data(const char* data, size_t size);

    // Check if vocabulary is loaded
    bool is_loaded() const noexcept;

    // Get vocabulary size
    size_t vocab_size() const noexcept;

    // Tokenize single text
    // max_length: maximum sequence length including [CLS] and [SEP]
    Result<TokenizedInput> tokenize(
        const std::string& text,
        uint32_t max_length = 256
    ) const;

    // Tokenize batch of texts
    Result<std::vector<TokenizedInput>> tokenize_batch(
        const std::vector<std::string>& texts,
        uint32_t max_length = 256
    ) const;

    // Convert token to ID
    Result<uint32_t> token_to_id(const std::string& token) const;

    // Convert ID to token
    Result<std::string> id_to_token(uint32_t id) const;

    // Get special token IDs
    uint32_t cls_token_id() const noexcept;
    uint32_t sep_token_id() const noexcept;
    uint32_t pad_token_id() const noexcept;
    uint32_t unk_token_id() const noexcept;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace embedder
```

---

## Embedder Interface

### `Embedding`

Type alias for embedding output.

```cpp
namespace embedder {

// Single embedding: 384-dimensional float vector
using Embedding = std::vector<float>;

// Batch of embeddings
using EmbeddingBatch = std::vector<Embedding>;

// Model constants
constexpr uint32_t EMBEDDING_DIM = 384;
constexpr uint32_t MAX_POSITION_EMBEDDINGS = 512;
constexpr uint32_t VOCAB_SIZE = 30522;
constexpr uint32_t NUM_LAYERS = 6;
constexpr uint32_t NUM_ATTENTION_HEADS = 12;
constexpr uint32_t INTERMEDIATE_SIZE = 1536;

} // namespace embedder
```

### `Embedder`

Main embedder class for generating sentence embeddings.

```cpp
namespace embedder {

class Embedder {
public:
    // Default constructor (model must be loaded separately)
    Embedder();

    // Constructor with configuration (loads model immediately)
    explicit Embedder(const EmbedderConfig& config);

    // Non-copyable
    Embedder(const Embedder&) = delete;
    Embedder& operator=(const Embedder&) = delete;

    // Movable
    Embedder(Embedder&&) noexcept;
    Embedder& operator=(Embedder&&) noexcept;

    ~Embedder();

    // Load model weights from file
    Result<void> load(const std::string& weights_path);

    // Load model weights from memory
    Result<void> load_from_memory(const char* data, size_t size);

    // Check if model is loaded and ready
    bool is_loaded() const noexcept;

    // Get embedding dimension (384 for MiniLM-L6-v2)
    uint32_t embedding_dim() const noexcept;

    // Get maximum sequence length
    uint32_t max_seq_length() const noexcept;

    // Generate embedding for single text
    Result<Embedding> embed(const std::string& text) const;

    // Generate embeddings for batch of texts
    Result<EmbeddingBatch> embed_batch(const std::vector<std::string>& texts) const;

    // Generate embedding from pre-tokenized input
    Result<Embedding> embed_tokenized(const TokenizedInput& input) const;

    // Generate embeddings from pre-tokenized batch
    Result<EmbeddingBatch> embed_tokenized_batch(
        const std::vector<TokenizedInput>& inputs
    ) const;

    // Access the underlying tokenizer
    const Tokenizer& tokenizer() const noexcept;

    // Get configuration
    const EmbedderConfig& config() const noexcept;

    // Get model metadata (from weight file)
    struct ModelMetadata {
        std::string model_name;
        std::string model_version;
        uint32_t embedding_dim;
        uint32_t vocab_size;
        uint32_t num_layers;
        uint32_t num_attention_heads;
    };
    Result<ModelMetadata> metadata() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace embedder
```

---

## Utility Functions

### Embedding Operations

```cpp
namespace embedder {

// Compute cosine similarity between two embeddings
// Returns value in range [-1, 1]
// Precondition: both embeddings must have the same dimension
float cosine_similarity(const Embedding& a, const Embedding& b);

// Compute cosine similarities between query and corpus
// Returns vector of similarities (same order as corpus)
std::vector<float> cosine_similarities(
    const Embedding& query,
    const EmbeddingBatch& corpus
);

// Find top-k most similar embeddings
// Returns pairs of (index, similarity) sorted by descending similarity
std::vector<std::pair<size_t, float>> find_top_k(
    const Embedding& query,
    const EmbeddingBatch& corpus,
    size_t k
);

// Normalize embedding to unit length (in-place)
void normalize(Embedding& embedding);

// Normalize embedding to unit length (copy)
Embedding normalized(const Embedding& embedding);

// Compute L2 norm of embedding
float norm(const Embedding& embedding);

// Compute dot product of two embeddings
float dot_product(const Embedding& a, const Embedding& b);

} // namespace embedder
```

### Version Information

```cpp
namespace embedder {

// Library version (semantic versioning)
constexpr uint32_t VERSION_MAJOR = 1;
constexpr uint32_t VERSION_MINOR = 0;
constexpr uint32_t VERSION_PATCH = 0;

// Get version string (e.g., "1.0.0")
std::string version();

// Get build information
std::string build_info();

} // namespace embedder
```

---

## Thread Safety

The following thread safety guarantees apply:

1. **`Tokenizer`**: Thread-safe for all `const` methods after construction/loading. Multiple threads may call `tokenize()` concurrently on the same instance.

2. **`Embedder`**: Thread-safe for all `const` methods after construction/loading. Multiple threads may call `embed()` concurrently on the same instance.

3. **Loading operations**: Not thread-safe. `load()` and `load_from_memory()` must not be called concurrently with any other method.

4. **Utility functions**: Thread-safe and re-entrant.

---

## Memory Management

1. All classes use RAII for resource management.
2. Move semantics are supported for efficient transfer of ownership.
3. No raw pointer ownership is exposed through public APIs.
4. Memory allocation failures are reported via `ErrorCode::ALLOCATION_FAILED`.

---

## Usage Example

```cpp
#include <embedder/embedder.h>
#include <iostream>

int main() {
    // Configure embedder
    embedder::EmbedderConfig config;
    config.weights_path = "model.weights";
    config.max_seq_length = 256;
    config.normalize_embeddings = true;

    // Create and load embedder
    embedder::Embedder emb(config);
    if (!emb.is_loaded()) {
        std::cerr << "Failed to load model" << std::endl;
        return 1;
    }

    // Generate embeddings
    auto result = emb.embed("This is a test sentence.");
    if (!result.ok()) {
        std::cerr << "Error: " << result.error_message() << std::endl;
        return 1;
    }

    auto embedding = result.value();
    std::cout << "Embedding dimension: " << embedding.size() << std::endl;

    // Batch embedding
    std::vector<std::string> texts = {
        "First sentence",
        "Second sentence",
        "Third sentence"
    };

    auto batch_result = emb.embed_batch(texts);
    if (batch_result.ok()) {
        auto embeddings = batch_result.value();

        // Find similar sentences
        auto similar = embedder::find_top_k(embeddings[0], embeddings, 2);
        for (auto& [idx, score] : similar) {
            std::cout << "Index " << idx << ": " << score << std::endl;
        }
    }

    return 0;
}
```

---

## Assumptions

1. **Endianness**: The library assumes little-endian byte order for weight files.
2. **Floating point**: IEEE 754 single-precision (32-bit) floats are used throughout.
3. **Character encoding**: All input text is assumed to be valid UTF-8.
4. **Memory model**: The library requires a flat memory model with byte-addressable memory.

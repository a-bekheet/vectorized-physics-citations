#ifndef CPP_EMBEDDER_MODEL_EMBEDDER_HPP
#define CPP_EMBEDDER_MODEL_EMBEDDER_HPP

#include "weights.hpp"
#include "layers.hpp"
#include "../math/tensor.hpp"
#include "../math/ops.hpp"
#include "../tokenizer/tokenizer.hpp"
#include <string>
#include <vector>
#include <memory>

namespace cpp_embedder {
namespace model {

// Model constant for embedding dimension (alias for clarity)
constexpr size_t EMBEDDING_DIM = HIDDEN_SIZE;  // 384

// Configuration for the embedder
struct EmbedderConfig {
    std::string weights_path;
    std::string vocab_path;  // Path to vocabulary file (required if not embedded)
    size_t max_seq_length = 256;
    bool normalize_embeddings = true;
};

// Sentence embedder implementing the full pipeline:
// Text -> Tokenize -> Embed -> Transformer -> Mean Pool -> Normalize
class Embedder {
public:
    // Construct with configuration
    explicit Embedder(const EmbedderConfig& config);

    // Construct with weight and vocab file paths
    Embedder(const std::string& weights_path, const std::string& vocab_path);

    // Non-copyable
    Embedder(const Embedder&) = delete;
    Embedder& operator=(const Embedder&) = delete;

    // Movable
    Embedder(Embedder&&) noexcept;
    Embedder& operator=(Embedder&&) noexcept;

    ~Embedder();

    // Embed a single text, returns 384-dim normalized vector
    std::vector<float> embed(const std::string& text) const;

    // Embed multiple texts, returns matrix of embeddings
    std::vector<std::vector<float>> embed_batch(const std::vector<std::string>& texts) const;

    // Get embedding dimension (384)
    size_t embedding_dim() const { return EMBEDDING_DIM; }

    // Get maximum sequence length
    size_t max_seq_length() const { return config_.max_seq_length; }

    // Check if model is loaded
    bool is_loaded() const { return is_loaded_; }

    // Access the underlying tokenizer
    const tokenizer::WordPieceTokenizer& tokenizer() const { return *tokenizer_; }

private:
    EmbedderConfig config_;
    bool is_loaded_ = false;

    // Model weights
    std::unique_ptr<ModelWeights> weights_;

    // Tokenizer
    std::unique_ptr<tokenizer::WordPieceTokenizer> tokenizer_;

    // Weight structures for layers (D3's types)
    std::unique_ptr<EmbeddingWeights> embedding_weights_;
    std::vector<std::unique_ptr<TransformerBlockWeights>> layer_weights_;

    // Layer instances (use D3's layer implementations)
    std::unique_ptr<Embedding> embedding_layer_;
    std::vector<std::unique_ptr<TransformerBlock>> transformer_blocks_;

    // Initialize model from weights
    void initialize();

    // Run transformer encoder stack
    // hidden_states: [seq_len, hidden_size]
    // Returns: [seq_len, hidden_size]
    math::Tensor encode(const math::Tensor& hidden_states) const;

    // Mean pooling over sequence
    // hidden_states: [seq_len, hidden_size]
    // attention_mask: [seq_len] - 1 for real tokens, 0 for padding
    // Returns: [hidden_size] vector
    math::Tensor mean_pool(const math::Tensor& hidden_states,
                          const std::vector<int>& attention_mask) const;

    // L2 normalize embedding
    void l2_normalize(std::vector<float>& embedding) const;
};

// Utility functions for working with embeddings

// Compute cosine similarity between two embeddings
float cosine_similarity(const std::vector<float>& a, const std::vector<float>& b);

// Compute dot product
float dot_product(const std::vector<float>& a, const std::vector<float>& b);

// L2 normalize a vector in-place
void normalize_embedding(std::vector<float>& embedding);

} // namespace model
} // namespace cpp_embedder

#endif // CPP_EMBEDDER_MODEL_EMBEDDER_HPP

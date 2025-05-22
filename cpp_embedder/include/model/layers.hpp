#ifndef CPP_EMBEDDER_MODEL_LAYERS_HPP
#define CPP_EMBEDDER_MODEL_LAYERS_HPP

#include "../math/tensor.hpp"
#include "../math/ops.hpp"
#include <cstddef>

namespace cpp_embedder {
namespace model {

// =============================================================================
// Model Constants (all-MiniLM-L6-v2)
// =============================================================================

constexpr std::size_t HIDDEN_SIZE = 384;
constexpr std::size_t INTERMEDIATE_SIZE = 1536;
constexpr std::size_t NUM_ATTENTION_HEADS = 12;
constexpr std::size_t HEAD_DIM = 32;  // HIDDEN_SIZE / NUM_ATTENTION_HEADS
constexpr std::size_t NUM_LAYERS = 6;
constexpr std::size_t MAX_POSITION_EMBEDDINGS = 512;
constexpr std::size_t VOCAB_SIZE = 30522;

// =============================================================================
// Weight Structures
// =============================================================================

/// Weights for a single attention layer
struct AttentionWeights {
    math::Tensor query_weight;   // (hidden_size, hidden_size)
    math::Tensor query_bias;     // (hidden_size,)
    math::Tensor key_weight;     // (hidden_size, hidden_size)
    math::Tensor key_bias;       // (hidden_size,)
    math::Tensor value_weight;   // (hidden_size, hidden_size)
    math::Tensor value_bias;     // (hidden_size,)
    math::Tensor output_weight;  // (hidden_size, hidden_size)
    math::Tensor output_bias;    // (hidden_size,)
};

/// Weights for a feed-forward network
struct FeedForwardWeights {
    math::Tensor intermediate_weight;  // (hidden_size, intermediate_size)
    math::Tensor intermediate_bias;    // (intermediate_size,)
    math::Tensor output_weight;        // (intermediate_size, hidden_size)
    math::Tensor output_bias;          // (hidden_size,)
};

/// Weights for layer normalization
struct LayerNormWeights {
    math::Tensor gamma;  // (hidden_size,)
    math::Tensor beta;   // (hidden_size,)
};

/// Complete weights for a transformer encoder block
struct TransformerBlockWeights {
    AttentionWeights attention;
    FeedForwardWeights ffn;
    LayerNormWeights attention_layer_norm;
    LayerNormWeights ffn_layer_norm;
};

/// Embedding weights
struct EmbeddingWeights {
    math::Tensor word_embeddings;      // (vocab_size, hidden_size)
    math::Tensor position_embeddings;  // (max_position, hidden_size)
    math::Tensor token_type_embeddings; // (2, hidden_size) for BERT
    LayerNormWeights layer_norm;
};

// =============================================================================
// Multi-Head Self-Attention
// =============================================================================

/// Multi-head self-attention layer
/// Implements: softmax(QK^T / sqrt(d_k)) * V
class MultiHeadAttention {
public:
    /// Construct with pre-loaded weights
    explicit MultiHeadAttention(const AttentionWeights& weights);

    /// Forward pass
    /// input: (seq_len, hidden_size)
    /// returns: (seq_len, hidden_size)
    math::Tensor forward(const math::Tensor& input) const;

private:
    const AttentionWeights& weights_;

    /// Split tensor into multiple heads
    /// input: (seq_len, hidden_size)
    /// returns: (num_heads, seq_len, head_dim)
    math::Tensor split_heads(const math::Tensor& x) const;

    /// Merge heads back
    /// input: (num_heads, seq_len, head_dim)
    /// returns: (seq_len, hidden_size)
    math::Tensor merge_heads(const math::Tensor& x) const;
};

// =============================================================================
// Feed-Forward Network
// =============================================================================

/// Position-wise feed-forward network
/// Implements: GELU(x * W1 + b1) * W2 + b2
class FeedForward {
public:
    /// Construct with pre-loaded weights
    explicit FeedForward(const FeedForwardWeights& weights);

    /// Forward pass
    /// input: (seq_len, hidden_size)
    /// returns: (seq_len, hidden_size)
    math::Tensor forward(const math::Tensor& input) const;

private:
    const FeedForwardWeights& weights_;
};

// =============================================================================
// Transformer Encoder Block
// =============================================================================

/// Single transformer encoder block
/// Implements: x + Attention(LayerNorm(x)), then x + FFN(LayerNorm(x))
/// Note: Uses pre-layer normalization (more stable)
class TransformerBlock {
public:
    /// Construct with pre-loaded weights
    explicit TransformerBlock(const TransformerBlockWeights& weights);

    /// Forward pass
    /// input: (seq_len, hidden_size)
    /// returns: (seq_len, hidden_size)
    math::Tensor forward(const math::Tensor& input) const;

private:
    MultiHeadAttention attention_;
    FeedForward ffn_;
    const TransformerBlockWeights& weights_;
};

// =============================================================================
// Positional Embeddings
// =============================================================================

/// Learned positional embeddings (BERT-style)
class PositionalEmbedding {
public:
    /// Construct with pre-loaded position embedding weights
    explicit PositionalEmbedding(const math::Tensor& position_embeddings);

    /// Get position embeddings for a sequence
    /// seq_len: length of the sequence
    /// returns: (seq_len, hidden_size)
    math::Tensor forward(std::size_t seq_len) const;

private:
    const math::Tensor& position_embeddings_;
};

// =============================================================================
// Embedding Layer
// =============================================================================

/// Combined embedding layer (word + position + token_type)
class Embedding {
public:
    /// Construct with pre-loaded weights
    explicit Embedding(const EmbeddingWeights& weights);

    /// Forward pass
    /// input_ids: (seq_len,) token IDs
    /// token_type_ids: (seq_len,) segment IDs (optional, defaults to 0)
    /// returns: (seq_len, hidden_size)
    math::Tensor forward(const std::vector<int>& input_ids,
                         const std::vector<int>& token_type_ids = {}) const;

private:
    const EmbeddingWeights& weights_;
    PositionalEmbedding pos_embedding_;
};

} // namespace model
} // namespace cpp_embedder

#endif // CPP_EMBEDDER_MODEL_LAYERS_HPP

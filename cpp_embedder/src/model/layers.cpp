#include "../../include/model/layers.hpp"
#include <cmath>
#include <stdexcept>

namespace cpp_embedder {
namespace model {

using math::Tensor;

// =============================================================================
// Multi-Head Self-Attention Implementation
// =============================================================================

MultiHeadAttention::MultiHeadAttention(const AttentionWeights& weights)
    : weights_(weights) {}

Tensor MultiHeadAttention::split_heads(const Tensor& x) const {
    // x: (seq_len, hidden_size)
    // output: (num_heads, seq_len, head_dim)

    std::size_t seq_len = x.dim(0);
    Tensor result(Tensor::Shape{NUM_ATTENTION_HEADS, seq_len, HEAD_DIM});

    for (std::size_t h = 0; h < NUM_ATTENTION_HEADS; ++h) {
        for (std::size_t s = 0; s < seq_len; ++s) {
            for (std::size_t d = 0; d < HEAD_DIM; ++d) {
                // Map from (seq_len, hidden_size) to (num_heads, seq_len, head_dim)
                // hidden_size index = h * HEAD_DIM + d
                result.at(h, s, d) = x.at(s, h * HEAD_DIM + d);
            }
        }
    }
    return result;
}

Tensor MultiHeadAttention::merge_heads(const Tensor& x) const {
    // x: (num_heads, seq_len, head_dim)
    // output: (seq_len, hidden_size)

    std::size_t seq_len = x.dim(1);
    Tensor result(Tensor::Shape{seq_len, HIDDEN_SIZE});

    for (std::size_t h = 0; h < NUM_ATTENTION_HEADS; ++h) {
        for (std::size_t s = 0; s < seq_len; ++s) {
            for (std::size_t d = 0; d < HEAD_DIM; ++d) {
                result.at(s, h * HEAD_DIM + d) = x.at(h, s, d);
            }
        }
    }
    return result;
}

Tensor MultiHeadAttention::forward(const Tensor& input) const {
    // input: (seq_len, hidden_size)
    std::size_t seq_len = input.dim(0);

    // =================================================================
    // Step 1: Linear projections for Q, K, V
    // Q = input @ W_q + b_q
    // K = input @ W_k + b_k
    // V = input @ W_v + b_v
    // =================================================================

    Tensor Q = math::matmul(input, weights_.query_weight);
    Tensor K = math::matmul(input, weights_.key_weight);
    Tensor V = math::matmul(input, weights_.value_weight);

    // Add biases (broadcast across sequence)
    for (std::size_t s = 0; s < seq_len; ++s) {
        for (std::size_t h = 0; h < HIDDEN_SIZE; ++h) {
            Q.at(s, h) += weights_.query_bias[h];
            K.at(s, h) += weights_.key_bias[h];
            V.at(s, h) += weights_.value_bias[h];
        }
    }

    // =================================================================
    // Step 2: Split into multiple heads
    // Reshape from (seq_len, hidden_size) to (num_heads, seq_len, head_dim)
    // =================================================================

    Tensor Q_heads = split_heads(Q);
    Tensor K_heads = split_heads(K);
    Tensor V_heads = split_heads(V);

    // =================================================================
    // Step 3: Scaled dot-product attention for each head
    // attention_scores = Q @ K^T / sqrt(head_dim)
    // attention_probs = softmax(attention_scores)
    // context = attention_probs @ V
    // =================================================================

    float scale = 1.0f / std::sqrt(static_cast<float>(HEAD_DIM));

    // Allocate output for all heads: (num_heads, seq_len, head_dim)
    Tensor context_heads(Tensor::Shape{NUM_ATTENTION_HEADS, seq_len, HEAD_DIM});

    for (std::size_t h = 0; h < NUM_ATTENTION_HEADS; ++h) {
        // Extract Q, K, V for this head: (seq_len, head_dim)
        Tensor Q_h(Tensor::Shape{seq_len, HEAD_DIM});
        Tensor K_h(Tensor::Shape{seq_len, HEAD_DIM});
        Tensor V_h(Tensor::Shape{seq_len, HEAD_DIM});

        for (std::size_t s = 0; s < seq_len; ++s) {
            for (std::size_t d = 0; d < HEAD_DIM; ++d) {
                Q_h.at(s, d) = Q_heads.at(h, s, d);
                K_h.at(s, d) = K_heads.at(h, s, d);
                V_h.at(s, d) = V_heads.at(h, s, d);
            }
        }

        // Compute attention scores: Q @ K^T -> (seq_len, seq_len)
        Tensor K_t = math::transpose(K_h);
        Tensor scores = math::matmul(Q_h, K_t);

        // Scale scores
        math::scale_inplace(scores, scale);

        // Apply softmax to get attention probabilities
        // softmax along last axis (each row sums to 1)
        Tensor probs = math::softmax(scores, -1);

        // Compute context: probs @ V -> (seq_len, head_dim)
        Tensor context = math::matmul(probs, V_h);

        // Store in output
        for (std::size_t s = 0; s < seq_len; ++s) {
            for (std::size_t d = 0; d < HEAD_DIM; ++d) {
                context_heads.at(h, s, d) = context.at(s, d);
            }
        }
    }

    // =================================================================
    // Step 4: Concatenate heads and apply output projection
    // output = concat(heads) @ W_o + b_o
    // =================================================================

    Tensor concatenated = merge_heads(context_heads);

    // Output projection
    Tensor output = math::matmul(concatenated, weights_.output_weight);

    // Add bias
    for (std::size_t s = 0; s < seq_len; ++s) {
        for (std::size_t h = 0; h < HIDDEN_SIZE; ++h) {
            output.at(s, h) += weights_.output_bias[h];
        }
    }

    return output;
}

// =============================================================================
// Feed-Forward Network Implementation
// =============================================================================

FeedForward::FeedForward(const FeedForwardWeights& weights)
    : weights_(weights) {}

Tensor FeedForward::forward(const Tensor& input) const {
    // input: (seq_len, hidden_size)
    std::size_t seq_len = input.dim(0);

    // =================================================================
    // Step 1: Expand to intermediate size
    // intermediate = GELU(input @ W1 + b1)
    // =================================================================

    Tensor intermediate = math::matmul(input, weights_.intermediate_weight);

    // Add bias
    for (std::size_t s = 0; s < seq_len; ++s) {
        for (std::size_t i = 0; i < INTERMEDIATE_SIZE; ++i) {
            intermediate.at(s, i) += weights_.intermediate_bias[i];
        }
    }

    // Apply GELU activation
    math::gelu_inplace(intermediate);

    // =================================================================
    // Step 2: Project back to hidden size
    // output = intermediate @ W2 + b2
    // =================================================================

    Tensor output = math::matmul(intermediate, weights_.output_weight);

    // Add bias
    for (std::size_t s = 0; s < seq_len; ++s) {
        for (std::size_t h = 0; h < HIDDEN_SIZE; ++h) {
            output.at(s, h) += weights_.output_bias[h];
        }
    }

    return output;
}

// =============================================================================
// Transformer Block Implementation
// =============================================================================

TransformerBlock::TransformerBlock(const TransformerBlockWeights& weights)
    : attention_(weights.attention),
      ffn_(weights.ffn),
      weights_(weights) {}

Tensor TransformerBlock::forward(const Tensor& input) const {
    // input: (seq_len, hidden_size)
    std::size_t seq_len = input.dim(0);

    // =================================================================
    // Step 1: Self-attention with residual connection
    // Note: BERT uses post-LN: output = LayerNorm(x + Attention(x))
    // =================================================================

    // Apply attention
    Tensor attn_output = attention_.forward(input);

    // Residual connection: x + Attention(x)
    Tensor residual1(Tensor::Shape{seq_len, HIDDEN_SIZE});
    for (std::size_t s = 0; s < seq_len; ++s) {
        for (std::size_t h = 0; h < HIDDEN_SIZE; ++h) {
            residual1.at(s, h) = input.at(s, h) + attn_output.at(s, h);
        }
    }

    // Apply layer normalization
    Tensor normed1 = math::layer_norm(
        residual1,
        weights_.attention_layer_norm.gamma,
        weights_.attention_layer_norm.beta
    );

    // =================================================================
    // Step 2: Feed-forward with residual connection
    // output = LayerNorm(x + FFN(x))
    // =================================================================

    // Apply feed-forward
    Tensor ffn_output = ffn_.forward(normed1);

    // Residual connection
    Tensor residual2(Tensor::Shape{seq_len, HIDDEN_SIZE});
    for (std::size_t s = 0; s < seq_len; ++s) {
        for (std::size_t h = 0; h < HIDDEN_SIZE; ++h) {
            residual2.at(s, h) = normed1.at(s, h) + ffn_output.at(s, h);
        }
    }

    // Apply layer normalization
    Tensor output = math::layer_norm(
        residual2,
        weights_.ffn_layer_norm.gamma,
        weights_.ffn_layer_norm.beta
    );

    return output;
}

// =============================================================================
// Positional Embedding Implementation
// =============================================================================

PositionalEmbedding::PositionalEmbedding(const Tensor& position_embeddings)
    : position_embeddings_(position_embeddings) {}

Tensor PositionalEmbedding::forward(std::size_t seq_len) const {
    if (seq_len > MAX_POSITION_EMBEDDINGS) {
        throw std::invalid_argument(
            "Sequence length exceeds maximum position embeddings");
    }

    // Extract position embeddings for positions 0 to seq_len-1
    // position_embeddings_: (max_position, hidden_size)
    // output: (seq_len, hidden_size)

    Tensor result(Tensor::Shape{seq_len, HIDDEN_SIZE});
    for (std::size_t p = 0; p < seq_len; ++p) {
        for (std::size_t h = 0; h < HIDDEN_SIZE; ++h) {
            result.at(p, h) = position_embeddings_.at(p, h);
        }
    }
    return result;
}

// =============================================================================
// Embedding Layer Implementation
// =============================================================================

Embedding::Embedding(const EmbeddingWeights& weights)
    : weights_(weights),
      pos_embedding_(weights.position_embeddings) {}

Tensor Embedding::forward(const std::vector<int>& input_ids,
                          const std::vector<int>& token_type_ids) const {
    std::size_t seq_len = input_ids.size();

    if (seq_len == 0) {
        throw std::invalid_argument("Empty input sequence");
    }

    if (seq_len > MAX_POSITION_EMBEDDINGS) {
        throw std::invalid_argument(
            "Sequence length exceeds maximum position embeddings");
    }

    // =================================================================
    // Step 1: Look up word embeddings
    // =================================================================

    Tensor embeddings(Tensor::Shape{seq_len, HIDDEN_SIZE});
    for (std::size_t s = 0; s < seq_len; ++s) {
        int token_id = input_ids[s];
        if (token_id < 0 || static_cast<std::size_t>(token_id) >= VOCAB_SIZE) {
            throw std::invalid_argument("Invalid token ID");
        }
        for (std::size_t h = 0; h < HIDDEN_SIZE; ++h) {
            embeddings.at(s, h) = weights_.word_embeddings.at(token_id, h);
        }
    }

    // =================================================================
    // Step 2: Add position embeddings
    // =================================================================

    Tensor pos_emb = pos_embedding_.forward(seq_len);
    for (std::size_t s = 0; s < seq_len; ++s) {
        for (std::size_t h = 0; h < HIDDEN_SIZE; ++h) {
            embeddings.at(s, h) += pos_emb.at(s, h);
        }
    }

    // =================================================================
    // Step 3: Add token type embeddings (segment embeddings)
    // =================================================================

    // Default to all zeros if not provided
    for (std::size_t s = 0; s < seq_len; ++s) {
        int type_id = 0;
        if (!token_type_ids.empty() && s < token_type_ids.size()) {
            type_id = token_type_ids[s];
        }
        if (type_id < 0 || type_id > 1) {
            throw std::invalid_argument("Invalid token type ID");
        }
        for (std::size_t h = 0; h < HIDDEN_SIZE; ++h) {
            embeddings.at(s, h) += weights_.token_type_embeddings.at(type_id, h);
        }
    }

    // =================================================================
    // Step 4: Apply layer normalization
    // =================================================================

    Tensor output = math::layer_norm(
        embeddings,
        weights_.layer_norm.gamma,
        weights_.layer_norm.beta
    );

    return output;
}

} // namespace model
} // namespace cpp_embedder

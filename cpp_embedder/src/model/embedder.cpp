#include "../../include/model/embedder.hpp"
#include <cmath>
#include <stdexcept>
#include <numeric>

namespace cpp_embedder {
namespace model {

using math::Tensor;
using tokenizer::Vocab;

// =============================================================================
// Embedder Implementation
// =============================================================================

Embedder::Embedder(const EmbedderConfig& config) : config_(config) {
    initialize();
}

Embedder::Embedder(const std::string& weights_path, const std::string& vocab_path) {
    config_.weights_path = weights_path;
    config_.vocab_path = vocab_path;
    initialize();
}

Embedder::Embedder(Embedder&&) noexcept = default;
Embedder& Embedder::operator=(Embedder&&) noexcept = default;
Embedder::~Embedder() = default;

void Embedder::initialize() {
    // Load weights from file
    weights_ = std::make_unique<ModelWeights>(WeightLoader::load(config_.weights_path));

    // Validate weights
    auto missing = WeightValidator::get_missing_tensors(*weights_, NUM_LAYERS);
    if (!missing.empty()) {
        std::string msg = "Missing tensors: ";
        for (size_t i = 0; i < missing.size() && i < 5; ++i) {
            msg += missing[i];
            if (i < missing.size() - 1) msg += ", ";
        }
        if (missing.size() > 5) {
            msg += "... and " + std::to_string(missing.size() - 5) + " more";
        }
        throw std::runtime_error(msg);
    }

    // Initialize tokenizer from vocab file
    if (!config_.vocab_path.empty()) {
        tokenizer_ = tokenizer::WordPieceTokenizer::from_vocab_file(config_.vocab_path);
        if (!tokenizer_) {
            throw std::runtime_error("Failed to load tokenizer from: " + config_.vocab_path);
        }
    } else if (weights_->has_vocabulary()) {
        // TODO: Support loading vocab from embedded weights
        throw std::runtime_error("Embedded vocabulary loading not yet implemented - use vocab_path in config");
    } else {
        throw std::runtime_error("No vocabulary source specified - set vocab_path in config");
    }

    // Build embedding weights structure (D3's type)
    embedding_weights_ = std::make_unique<EmbeddingWeights>();
    embedding_weights_->word_embeddings = weights_->get("embeddings.word_embeddings.weight");
    embedding_weights_->position_embeddings = weights_->get("embeddings.position_embeddings.weight");
    embedding_weights_->token_type_embeddings = weights_->get("embeddings.token_type_embeddings.weight");
    embedding_weights_->layer_norm.gamma = weights_->get("embeddings.LayerNorm.weight");
    embedding_weights_->layer_norm.beta = weights_->get("embeddings.LayerNorm.bias");

    // Create embedding layer
    embedding_layer_ = std::make_unique<Embedding>(*embedding_weights_);

    // Build transformer block weights and layers
    for (size_t i = 0; i < NUM_LAYERS; ++i) {
        std::string prefix = "encoder.layer." + std::to_string(i) + ".";

        auto block_weights = std::make_unique<TransformerBlockWeights>();

        // Attention weights
        block_weights->attention.query_weight = weights_->get(prefix + "attention.self.query.weight");
        block_weights->attention.query_bias = weights_->get(prefix + "attention.self.query.bias");
        block_weights->attention.key_weight = weights_->get(prefix + "attention.self.key.weight");
        block_weights->attention.key_bias = weights_->get(prefix + "attention.self.key.bias");
        block_weights->attention.value_weight = weights_->get(prefix + "attention.self.value.weight");
        block_weights->attention.value_bias = weights_->get(prefix + "attention.self.value.bias");
        block_weights->attention.output_weight = weights_->get(prefix + "attention.output.dense.weight");
        block_weights->attention.output_bias = weights_->get(prefix + "attention.output.dense.bias");

        // Attention layer norm
        block_weights->attention_layer_norm.gamma = weights_->get(prefix + "attention.output.LayerNorm.weight");
        block_weights->attention_layer_norm.beta = weights_->get(prefix + "attention.output.LayerNorm.bias");

        // Feed-forward weights
        block_weights->ffn.intermediate_weight = weights_->get(prefix + "intermediate.dense.weight");
        block_weights->ffn.intermediate_bias = weights_->get(prefix + "intermediate.dense.bias");
        block_weights->ffn.output_weight = weights_->get(prefix + "output.dense.weight");
        block_weights->ffn.output_bias = weights_->get(prefix + "output.dense.bias");

        // FFN layer norm
        block_weights->ffn_layer_norm.gamma = weights_->get(prefix + "output.LayerNorm.weight");
        block_weights->ffn_layer_norm.beta = weights_->get(prefix + "output.LayerNorm.bias");

        // Create transformer block
        transformer_blocks_.push_back(std::make_unique<TransformerBlock>(*block_weights));
        layer_weights_.push_back(std::move(block_weights));
    }

    is_loaded_ = true;
}

std::vector<float> Embedder::embed(const std::string& text) const {
    if (!is_loaded_) {
        throw std::runtime_error("Model not loaded");
    }
    if (!tokenizer_) {
        throw std::runtime_error("Tokenizer not initialized");
    }

    // Tokenize text
    auto token_ids = tokenizer_->encode(text, config_.max_seq_length);

    // Create attention mask (1 for real tokens, 0 for padding)
    std::vector<int> attention_mask(token_ids.size(), 1);
    Vocab::TokenId pad_id = tokenizer_->vocab().pad_id();
    for (size_t i = 0; i < token_ids.size(); ++i) {
        if (token_ids[i] == pad_id) {
            attention_mask[i] = 0;
        }
    }

    // Convert token IDs to int vector for embedding layer
    std::vector<int> input_ids(token_ids.begin(), token_ids.end());

    // Create embeddings using D3's Embedding layer
    Tensor hidden_states = embedding_layer_->forward(input_ids);

    // Run through transformer encoder
    hidden_states = encode(hidden_states);

    // Mean pooling
    Tensor pooled = mean_pool(hidden_states, attention_mask);

    // Convert to vector
    std::vector<float> embedding(pooled.data(), pooled.data() + pooled.size());

    // Normalize if configured
    if (config_.normalize_embeddings) {
        l2_normalize(embedding);
    }

    return embedding;
}

std::vector<std::vector<float>> Embedder::embed_batch(const std::vector<std::string>& texts) const {
    std::vector<std::vector<float>> embeddings;
    embeddings.reserve(texts.size());

    for (const auto& text : texts) {
        embeddings.push_back(embed(text));
    }

    return embeddings;
}

Tensor Embedder::encode(const Tensor& hidden_states) const {
    Tensor current = hidden_states;

    for (const auto& block : transformer_blocks_) {
        current = block->forward(current);
    }

    return current;
}

Tensor Embedder::mean_pool(const Tensor& hidden_states, const std::vector<int>& attention_mask) const {
    // hidden_states: [seq_len, hidden_size]
    // attention_mask: [seq_len]
    // output: [hidden_size]

    size_t seq_len = hidden_states.dim(0);
    size_t hidden_size = hidden_states.dim(1);

    Tensor pooled({hidden_size}, 0.0f);

    // Count non-padding tokens
    float token_count = 0.0f;
    for (size_t i = 0; i < attention_mask.size(); ++i) {
        token_count += attention_mask[i];
    }

    if (token_count == 0.0f) {
        throw std::runtime_error("All tokens are padding");
    }

    // Sum embeddings of non-padding tokens
    for (size_t s = 0; s < seq_len; ++s) {
        if (s < attention_mask.size() && attention_mask[s] == 1) {
            for (size_t h = 0; h < hidden_size; ++h) {
                pooled[h] += hidden_states.at(s, h);
            }
        }
    }

    // Divide by count to get mean
    for (size_t h = 0; h < hidden_size; ++h) {
        pooled[h] /= token_count;
    }

    return pooled;
}

void Embedder::l2_normalize(std::vector<float>& embedding) const {
    float norm = 0.0f;
    for (float v : embedding) {
        norm += v * v;
    }
    norm = std::sqrt(norm);

    if (norm > 1e-12f) {
        for (float& v : embedding) {
            v /= norm;
        }
    }
}

// =============================================================================
// Utility Functions
// =============================================================================

float cosine_similarity(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Embedding dimensions must match");
    }

    float dot = 0.0f;
    float norm_a = 0.0f;
    float norm_b = 0.0f;

    for (size_t i = 0; i < a.size(); ++i) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    norm_a = std::sqrt(norm_a);
    norm_b = std::sqrt(norm_b);

    if (norm_a < 1e-12f || norm_b < 1e-12f) {
        return 0.0f;
    }

    return dot / (norm_a * norm_b);
}

float dot_product(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vector dimensions must match");
    }

    float result = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        result += a[i] * b[i];
    }
    return result;
}

void normalize_embedding(std::vector<float>& embedding) {
    float norm = 0.0f;
    for (float v : embedding) {
        norm += v * v;
    }
    norm = std::sqrt(norm);

    if (norm > 1e-12f) {
        for (float& v : embedding) {
            v /= norm;
        }
    }
}

} // namespace model
} // namespace cpp_embedder

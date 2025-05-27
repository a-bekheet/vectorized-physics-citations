#include "../include/model/layers.hpp"
#include <iostream>
#include <cmath>
#include <cassert>
#include <random>

using namespace cpp_embedder;
using namespace cpp_embedder::model;
using namespace cpp_embedder::math;

// =============================================================================
// Test Utilities
// =============================================================================

namespace {

constexpr float EPSILON = 1e-5f;

bool approx_equal(float a, float b, float eps = EPSILON) {
    return std::abs(a - b) < eps;
}

// Create a random tensor with values in [-1, 1]
Tensor random_tensor(const Tensor::Shape& shape, unsigned seed = 42) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    Tensor::size_type total_size = 1;
    for (auto dim : shape) {
        total_size *= dim;
    }

    std::vector<float> data(total_size);
    for (auto& val : data) {
        val = dist(gen);
    }

    return Tensor(shape, std::move(data));
}

// Create attention weights for testing
AttentionWeights create_test_attention_weights(unsigned seed = 42) {
    AttentionWeights weights;

    // Scale down for numerical stability in tests
    float scale = 0.1f;

    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-scale, scale);

    auto fill_tensor = [&](Tensor& t) {
        for (Tensor::size_type i = 0; i < t.size(); ++i) {
            t[i] = dist(gen);
        }
    };

    weights.query_weight = Tensor(Tensor::Shape{HIDDEN_SIZE, HIDDEN_SIZE});
    weights.query_bias = Tensor(Tensor::Shape{HIDDEN_SIZE});
    weights.key_weight = Tensor(Tensor::Shape{HIDDEN_SIZE, HIDDEN_SIZE});
    weights.key_bias = Tensor(Tensor::Shape{HIDDEN_SIZE});
    weights.value_weight = Tensor(Tensor::Shape{HIDDEN_SIZE, HIDDEN_SIZE});
    weights.value_bias = Tensor(Tensor::Shape{HIDDEN_SIZE});
    weights.output_weight = Tensor(Tensor::Shape{HIDDEN_SIZE, HIDDEN_SIZE});
    weights.output_bias = Tensor(Tensor::Shape{HIDDEN_SIZE});

    fill_tensor(weights.query_weight);
    fill_tensor(weights.query_bias);
    fill_tensor(weights.key_weight);
    fill_tensor(weights.key_bias);
    fill_tensor(weights.value_weight);
    fill_tensor(weights.value_bias);
    fill_tensor(weights.output_weight);
    fill_tensor(weights.output_bias);

    return weights;
}

// Create feed-forward weights for testing
FeedForwardWeights create_test_ffn_weights(unsigned seed = 43) {
    FeedForwardWeights weights;

    float scale = 0.1f;
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-scale, scale);

    auto fill_tensor = [&](Tensor& t) {
        for (Tensor::size_type i = 0; i < t.size(); ++i) {
            t[i] = dist(gen);
        }
    };

    weights.intermediate_weight = Tensor(Tensor::Shape{HIDDEN_SIZE, INTERMEDIATE_SIZE});
    weights.intermediate_bias = Tensor(Tensor::Shape{INTERMEDIATE_SIZE});
    weights.output_weight = Tensor(Tensor::Shape{INTERMEDIATE_SIZE, HIDDEN_SIZE});
    weights.output_bias = Tensor(Tensor::Shape{HIDDEN_SIZE});

    fill_tensor(weights.intermediate_weight);
    fill_tensor(weights.intermediate_bias);
    fill_tensor(weights.output_weight);
    fill_tensor(weights.output_bias);

    return weights;
}

// Create layer norm weights for testing
LayerNormWeights create_test_layer_norm_weights(unsigned seed = 44) {
    LayerNormWeights weights;

    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(0.8f, 1.2f);
    std::uniform_real_distribution<float> dist_beta(-0.1f, 0.1f);

    weights.gamma = Tensor(Tensor::Shape{HIDDEN_SIZE});
    weights.beta = Tensor(Tensor::Shape{HIDDEN_SIZE});

    for (Tensor::size_type i = 0; i < HIDDEN_SIZE; ++i) {
        weights.gamma[i] = dist(gen);
        weights.beta[i] = dist_beta(gen);
    }

    return weights;
}

// Create transformer block weights
TransformerBlockWeights create_test_block_weights(unsigned seed = 45) {
    TransformerBlockWeights weights;
    weights.attention = create_test_attention_weights(seed);
    weights.ffn = create_test_ffn_weights(seed + 1);
    weights.attention_layer_norm = create_test_layer_norm_weights(seed + 2);
    weights.ffn_layer_norm = create_test_layer_norm_weights(seed + 3);
    return weights;
}

void print_test_result(const std::string& test_name, bool passed) {
    std::cout << (passed ? "[PASS] " : "[FAIL] ") << test_name << std::endl;
}

} // anonymous namespace

// =============================================================================
// Multi-Head Attention Tests
// =============================================================================

void test_attention_output_shape() {
    std::cout << "\n--- Multi-Head Attention Tests ---\n";

    auto weights = create_test_attention_weights();
    MultiHeadAttention attention(weights);

    // Test with sequence length 5
    Tensor input = random_tensor({5, HIDDEN_SIZE});
    Tensor output = attention.forward(input);

    bool passed = (output.ndim() == 2 &&
                   output.dim(0) == 5 &&
                   output.dim(1) == HIDDEN_SIZE);

    print_test_result("Attention output shape (seq_len=5)", passed);
    assert(passed);
}

void test_attention_single_token() {
    auto weights = create_test_attention_weights();
    MultiHeadAttention attention(weights);

    // Test with single token
    Tensor input = random_tensor({1, HIDDEN_SIZE});
    Tensor output = attention.forward(input);

    bool passed = (output.dim(0) == 1 && output.dim(1) == HIDDEN_SIZE);
    print_test_result("Attention with single token", passed);
    assert(passed);
}

void test_attention_deterministic() {
    auto weights = create_test_attention_weights();
    MultiHeadAttention attention(weights);

    Tensor input = random_tensor({3, HIDDEN_SIZE}, 123);

    Tensor output1 = attention.forward(input);
    Tensor output2 = attention.forward(input);

    bool passed = true;
    for (Tensor::size_type i = 0; i < output1.size(); ++i) {
        if (!approx_equal(output1[i], output2[i])) {
            passed = false;
            break;
        }
    }

    print_test_result("Attention is deterministic", passed);
    assert(passed);
}

// =============================================================================
// Feed-Forward Network Tests
// =============================================================================

void test_ffn_output_shape() {
    std::cout << "\n--- Feed-Forward Network Tests ---\n";

    auto weights = create_test_ffn_weights();
    FeedForward ffn(weights);

    Tensor input = random_tensor({7, HIDDEN_SIZE});
    Tensor output = ffn.forward(input);

    bool passed = (output.ndim() == 2 &&
                   output.dim(0) == 7 &&
                   output.dim(1) == HIDDEN_SIZE);

    print_test_result("FFN output shape (seq_len=7)", passed);
    assert(passed);
}

void test_ffn_intermediate_expansion() {
    auto weights = create_test_ffn_weights();
    FeedForward ffn(weights);

    // Verify weights have correct intermediate size
    bool passed = (weights.intermediate_weight.dim(1) == INTERMEDIATE_SIZE &&
                   weights.output_weight.dim(0) == INTERMEDIATE_SIZE);

    print_test_result("FFN intermediate expansion (4x)", passed);
    assert(passed);
}

// =============================================================================
// Transformer Block Tests
// =============================================================================

void test_block_output_shape() {
    std::cout << "\n--- Transformer Block Tests ---\n";

    auto weights = create_test_block_weights();
    TransformerBlock block(weights);

    Tensor input = random_tensor({10, HIDDEN_SIZE});
    Tensor output = block.forward(input);

    bool passed = (output.ndim() == 2 &&
                   output.dim(0) == 10 &&
                   output.dim(1) == HIDDEN_SIZE);

    print_test_result("Block output shape (seq_len=10)", passed);
    assert(passed);
}

void test_block_residual_connection() {
    auto weights = create_test_block_weights();
    TransformerBlock block(weights);

    // Create input with moderate values
    Tensor input = random_tensor({4, HIDDEN_SIZE}, 999);

    Tensor output = block.forward(input);

    // Output should not be zero (residual connections prevent vanishing)
    float output_norm = 0.0f;
    for (Tensor::size_type i = 0; i < output.size(); ++i) {
        output_norm += output[i] * output[i];
    }
    output_norm = std::sqrt(output_norm);

    bool passed = output_norm > 0.1f;
    print_test_result("Block has non-zero output (residual works)", passed);
    assert(passed);
}

// =============================================================================
// Positional Embedding Tests
// =============================================================================

void test_positional_embedding_shape() {
    std::cout << "\n--- Positional Embedding Tests ---\n";

    Tensor pos_emb_weights = random_tensor({MAX_POSITION_EMBEDDINGS, HIDDEN_SIZE});
    PositionalEmbedding pos_emb(pos_emb_weights);

    Tensor output = pos_emb.forward(15);

    bool passed = (output.ndim() == 2 &&
                   output.dim(0) == 15 &&
                   output.dim(1) == HIDDEN_SIZE);

    print_test_result("Position embedding shape (seq_len=15)", passed);
    assert(passed);
}

void test_positional_embedding_max_length() {
    Tensor pos_emb_weights = random_tensor({MAX_POSITION_EMBEDDINGS, HIDDEN_SIZE});
    PositionalEmbedding pos_emb(pos_emb_weights);

    // Should work at max length
    Tensor output = pos_emb.forward(MAX_POSITION_EMBEDDINGS);

    bool passed = output.dim(0) == MAX_POSITION_EMBEDDINGS;
    print_test_result("Position embedding at max length", passed);
    assert(passed);
}

void test_positional_embedding_exceeds_max() {
    Tensor pos_emb_weights = random_tensor({MAX_POSITION_EMBEDDINGS, HIDDEN_SIZE});
    PositionalEmbedding pos_emb(pos_emb_weights);

    bool threw_exception = false;
    try {
        pos_emb.forward(MAX_POSITION_EMBEDDINGS + 1);
    } catch (const std::invalid_argument&) {
        threw_exception = true;
    }

    print_test_result("Position embedding rejects exceeding max", threw_exception);
    assert(threw_exception);
}

// =============================================================================
// Embedding Layer Tests
// =============================================================================

void test_embedding_basic() {
    std::cout << "\n--- Embedding Layer Tests ---\n";

    EmbeddingWeights weights;
    weights.word_embeddings = random_tensor({VOCAB_SIZE, HIDDEN_SIZE}, 100);
    weights.position_embeddings = random_tensor({MAX_POSITION_EMBEDDINGS, HIDDEN_SIZE}, 101);
    weights.token_type_embeddings = random_tensor({2, HIDDEN_SIZE}, 102);
    weights.layer_norm = create_test_layer_norm_weights(103);

    Embedding emb(weights);

    std::vector<int> input_ids = {101, 2054, 2003, 2023, 102};  // [CLS] what is this [SEP]
    Tensor output = emb.forward(input_ids);

    bool passed = (output.ndim() == 2 &&
                   output.dim(0) == 5 &&
                   output.dim(1) == HIDDEN_SIZE);

    print_test_result("Embedding output shape", passed);
    assert(passed);
}

void test_embedding_with_token_types() {
    EmbeddingWeights weights;
    weights.word_embeddings = random_tensor({VOCAB_SIZE, HIDDEN_SIZE}, 200);
    weights.position_embeddings = random_tensor({MAX_POSITION_EMBEDDINGS, HIDDEN_SIZE}, 201);
    weights.token_type_embeddings = random_tensor({2, HIDDEN_SIZE}, 202);
    weights.layer_norm = create_test_layer_norm_weights(203);

    Embedding emb(weights);

    std::vector<int> input_ids = {101, 2054, 102, 2023, 102};
    std::vector<int> token_type_ids = {0, 0, 0, 1, 1};

    Tensor output = emb.forward(input_ids, token_type_ids);

    bool passed = output.dim(0) == 5;
    print_test_result("Embedding with token types", passed);
    assert(passed);
}

void test_embedding_invalid_token() {
    EmbeddingWeights weights;
    weights.word_embeddings = random_tensor({VOCAB_SIZE, HIDDEN_SIZE}, 300);
    weights.position_embeddings = random_tensor({MAX_POSITION_EMBEDDINGS, HIDDEN_SIZE}, 301);
    weights.token_type_embeddings = random_tensor({2, HIDDEN_SIZE}, 302);
    weights.layer_norm = create_test_layer_norm_weights(303);

    Embedding emb(weights);

    std::vector<int> input_ids = {101, 999999, 102};  // Invalid token ID

    bool threw_exception = false;
    try {
        emb.forward(input_ids);
    } catch (const std::invalid_argument&) {
        threw_exception = true;
    }

    print_test_result("Embedding rejects invalid token ID", threw_exception);
    assert(threw_exception);
}

// =============================================================================
// Integration Tests
// =============================================================================

void test_full_forward_pass() {
    std::cout << "\n--- Integration Tests ---\n";

    // Create embeddings
    EmbeddingWeights emb_weights;
    emb_weights.word_embeddings = random_tensor({VOCAB_SIZE, HIDDEN_SIZE}, 400);
    emb_weights.position_embeddings = random_tensor({MAX_POSITION_EMBEDDINGS, HIDDEN_SIZE}, 401);
    emb_weights.token_type_embeddings = random_tensor({2, HIDDEN_SIZE}, 402);
    emb_weights.layer_norm = create_test_layer_norm_weights(403);

    Embedding emb(emb_weights);

    // Create a single transformer block
    auto block_weights = create_test_block_weights(500);
    TransformerBlock block(block_weights);

    // Forward pass: tokens -> embeddings -> transformer block
    std::vector<int> input_ids = {101, 7592, 1010, 2088, 999, 102};  // [CLS] hello , world ! [SEP]

    Tensor embeddings = emb.forward(input_ids);
    Tensor output = block.forward(embeddings);

    bool passed = (output.dim(0) == 6 && output.dim(1) == HIDDEN_SIZE);
    print_test_result("Full forward pass (embedding + block)", passed);
    assert(passed);

    // Verify output is not all zeros or NaN
    bool has_valid_values = true;
    bool has_nonzero = false;
    for (Tensor::size_type i = 0; i < output.size(); ++i) {
        if (std::isnan(output[i]) || std::isinf(output[i])) {
            has_valid_values = false;
            break;
        }
        if (std::abs(output[i]) > 1e-6f) {
            has_nonzero = true;
        }
    }

    print_test_result("Output has valid values (no NaN/Inf)", has_valid_values);
    print_test_result("Output has non-zero values", has_nonzero);
    assert(has_valid_values && has_nonzero);
}

void test_multiple_blocks() {
    // Simulate stacking multiple transformer blocks
    auto block_weights1 = create_test_block_weights(600);
    auto block_weights2 = create_test_block_weights(700);

    TransformerBlock block1(block_weights1);
    TransformerBlock block2(block_weights2);

    Tensor input = random_tensor({8, HIDDEN_SIZE}, 888);

    Tensor hidden = block1.forward(input);
    Tensor output = block2.forward(hidden);

    bool passed = (output.dim(0) == 8 && output.dim(1) == HIDDEN_SIZE);
    print_test_result("Stacked transformer blocks", passed);
    assert(passed);
}

// =============================================================================
// Main
// =============================================================================

int main() {
    std::cout << "========================================\n";
    std::cout << "  Transformer Layer Tests\n";
    std::cout << "  Model: all-MiniLM-L6-v2 architecture\n";
    std::cout << "========================================\n";

    // Multi-Head Attention tests
    test_attention_output_shape();
    test_attention_single_token();
    test_attention_deterministic();

    // Feed-Forward Network tests
    test_ffn_output_shape();
    test_ffn_intermediate_expansion();

    // Transformer Block tests
    test_block_output_shape();
    test_block_residual_connection();

    // Positional Embedding tests
    test_positional_embedding_shape();
    test_positional_embedding_max_length();
    test_positional_embedding_exceeds_max();

    // Embedding Layer tests
    test_embedding_basic();
    test_embedding_with_token_types();
    test_embedding_invalid_token();

    // Integration tests
    test_full_forward_pass();
    test_multiple_blocks();

    std::cout << "\n========================================\n";
    std::cout << "  All tests passed!\n";
    std::cout << "========================================\n";

    return 0;
}

/**
 * @file c_api.cpp
 * @brief C API implementation for cpp_embedder library
 *
 * This file implements the C API defined in c_api.h by wrapping the
 * underlying C++ Embedder class. It handles:
 * - Memory management between C and C++
 * - Error handling and thread-local error state
 * - Type conversions between C and C++ types
 */

#include "bindings/c_api.h"
#include "model/embedder.hpp"
#include "tokenizer/tokenizer.hpp"

#include <cmath>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace {

// Thread-local error state
thread_local std::string g_error_message;
thread_local CppEmbedderError g_error_code = CPP_EMBEDDER_OK;

void set_error(CppEmbedderError code, const std::string& message) {
    g_error_code = code;
    g_error_message = message;
}

void clear_error() {
    g_error_code = CPP_EMBEDDER_OK;
    g_error_message.clear();
}

// Version information
const char* VERSION_STRING = "1.0.0";
const char* BUILD_INFO_STRING = "cpp_embedder v1.0.0 (C++ 17)";

// Wrapper struct to hold embedder configuration and state
struct EmbedderWrapper {
    cpp_embedder::model::EmbedderConfig config;
    std::unique_ptr<cpp_embedder::model::Embedder> embedder;
    bool should_normalize = true;

    EmbedderWrapper() = default;
    ~EmbedderWrapper() = default;
};

} // anonymous namespace

// =============================================================================
// C API Implementation
// =============================================================================

extern "C" {

// Version Information
CPP_EMBEDDER_API const char* embedder_version(void) {
    return VERSION_STRING;
}

CPP_EMBEDDER_API const char* embedder_build_info(void) {
    return BUILD_INFO_STRING;
}

// Error Handling
CPP_EMBEDDER_API const char* embedder_get_error(void) {
    return g_error_message.c_str();
}

CPP_EMBEDDER_API CppEmbedderError embedder_get_error_code(void) {
    return g_error_code;
}

CPP_EMBEDDER_API void embedder_clear_error(void) {
    clear_error();
}

// Embedder Lifecycle
CPP_EMBEDDER_API EmbedderHandle embedder_create(const char* weights_path) {
    return embedder_create_with_config(weights_path, 0, 1, 0);
}

CPP_EMBEDDER_API EmbedderHandle embedder_create_with_config(
    const char* weights_path,
    uint32_t max_seq_length,
    int normalize,
    uint32_t num_threads
) {
    clear_error();

    try {
        auto wrapper = new EmbedderWrapper();

        wrapper->config.max_seq_length = max_seq_length > 0 ? max_seq_length : 256;
        wrapper->config.normalize_embeddings = normalize != 0;
        wrapper->should_normalize = normalize != 0;

        if (weights_path && weights_path[0] != '\0') {
            wrapper->config.weights_path = weights_path;

            // Derive vocab path from weights path (assume .vocab extension)
            std::string vocab_path = std::string(weights_path);
            size_t dot_pos = vocab_path.rfind('.');
            if (dot_pos != std::string::npos) {
                vocab_path = vocab_path.substr(0, dot_pos) + ".vocab";
            } else {
                vocab_path += ".vocab";
            }
            wrapper->config.vocab_path = vocab_path;

            try {
                wrapper->embedder = std::make_unique<cpp_embedder::model::Embedder>(
                    wrapper->config
                );
            } catch (const std::runtime_error& e) {
                // Try without vocab path if that failed
                wrapper->config.vocab_path = "";
                set_error(CPP_EMBEDDER_ERROR_FILE_NOT_FOUND,
                          "Failed to load model: " + std::string(e.what()));
                delete wrapper;
                return nullptr;
            }
        }

        return static_cast<EmbedderHandle>(wrapper);

    } catch (const std::bad_alloc&) {
        set_error(CPP_EMBEDDER_ERROR_ALLOCATION_FAILED, "Memory allocation failed");
        return nullptr;
    } catch (const std::exception& e) {
        set_error(CPP_EMBEDDER_ERROR_UNKNOWN, e.what());
        return nullptr;
    }
}

CPP_EMBEDDER_API EmbedderHandle embedder_create_with_vocab(
    const char* weights_path,
    const char* vocab_path
) {
    clear_error();

    try {
        auto wrapper = new EmbedderWrapper();

        wrapper->config.max_seq_length = 256;
        wrapper->config.normalize_embeddings = true;
        wrapper->should_normalize = true;

        if (weights_path && weights_path[0] != '\0') {
            wrapper->config.weights_path = weights_path;
        }

        if (vocab_path && vocab_path[0] != '\0') {
            wrapper->config.vocab_path = vocab_path;
        }

        try {
            wrapper->embedder = std::make_unique<cpp_embedder::model::Embedder>(
                wrapper->config
            );
        } catch (const std::exception& e) {
            set_error(CPP_EMBEDDER_ERROR_FILE_NOT_FOUND,
                      "Failed to load model: " + std::string(e.what()));
            delete wrapper;
            return nullptr;
        }

        return static_cast<EmbedderHandle>(wrapper);

    } catch (const std::bad_alloc&) {
        set_error(CPP_EMBEDDER_ERROR_ALLOCATION_FAILED, "Memory allocation failed");
        return nullptr;
    } catch (const std::exception& e) {
        set_error(CPP_EMBEDDER_ERROR_UNKNOWN, e.what());
        return nullptr;
    }
}

CPP_EMBEDDER_API CppEmbedderError embedder_load(EmbedderHandle handle, const char* weights_path) {
    clear_error();

    if (!handle) {
        set_error(CPP_EMBEDDER_ERROR_NULL_POINTER, "Null embedder handle");
        return CPP_EMBEDDER_ERROR_NULL_POINTER;
    }

    if (!weights_path) {
        set_error(CPP_EMBEDDER_ERROR_NULL_POINTER, "Null weights path");
        return CPP_EMBEDDER_ERROR_NULL_POINTER;
    }

    try {
        auto wrapper = static_cast<EmbedderWrapper*>(handle);
        wrapper->config.weights_path = weights_path;

        // Derive vocab path from weights path
        std::string vocab_path = std::string(weights_path);
        size_t dot_pos = vocab_path.rfind('.');
        if (dot_pos != std::string::npos) {
            vocab_path = vocab_path.substr(0, dot_pos) + ".vocab";
        } else {
            vocab_path += ".vocab";
        }
        wrapper->config.vocab_path = vocab_path;

        wrapper->embedder = std::make_unique<cpp_embedder::model::Embedder>(
            wrapper->config
        );

        return CPP_EMBEDDER_OK;

    } catch (const std::runtime_error& e) {
        set_error(CPP_EMBEDDER_ERROR_FILE_NOT_FOUND,
                  "Failed to load weights: " + std::string(e.what()));
        return CPP_EMBEDDER_ERROR_FILE_NOT_FOUND;
    } catch (const std::exception& e) {
        set_error(CPP_EMBEDDER_ERROR_UNKNOWN, e.what());
        return CPP_EMBEDDER_ERROR_UNKNOWN;
    }
}

CPP_EMBEDDER_API void embedder_destroy(EmbedderHandle handle) {
    if (handle) {
        delete static_cast<EmbedderWrapper*>(handle);
    }
}

CPP_EMBEDDER_API int embedder_is_loaded(EmbedderHandle handle) {
    if (!handle) return 0;
    auto wrapper = static_cast<EmbedderWrapper*>(handle);
    return (wrapper->embedder && wrapper->embedder->is_loaded()) ? 1 : 0;
}

// Embedder Properties
CPP_EMBEDDER_API uint32_t embedder_get_dim(EmbedderHandle handle) {
    if (!handle) return 0;
    auto wrapper = static_cast<EmbedderWrapper*>(handle);
    if (!wrapper->embedder) return CPP_EMBEDDER_DIM;
    return static_cast<uint32_t>(wrapper->embedder->embedding_dim());
}

CPP_EMBEDDER_API uint32_t embedder_get_max_seq_length(EmbedderHandle handle) {
    if (!handle) return 0;
    auto wrapper = static_cast<EmbedderWrapper*>(handle);
    if (!wrapper->embedder) return static_cast<uint32_t>(wrapper->config.max_seq_length);
    return static_cast<uint32_t>(wrapper->embedder->max_seq_length());
}

CPP_EMBEDDER_API uint32_t embedder_get_vocab_size(EmbedderHandle handle) {
    if (!handle) return 0;
    auto wrapper = static_cast<EmbedderWrapper*>(handle);
    if (!wrapper->embedder) return CPP_EMBEDDER_VOCAB_SIZE;
    return static_cast<uint32_t>(wrapper->embedder->tokenizer().vocab().size());
}

// Single Text Embedding
CPP_EMBEDDER_API CppEmbedderError embedder_embed(
    EmbedderHandle handle,
    const char* text,
    float* output,
    uint32_t output_size
) {
    if (!handle) {
        set_error(CPP_EMBEDDER_ERROR_NULL_POINTER, "Null embedder handle");
        return CPP_EMBEDDER_ERROR_NULL_POINTER;
    }

    auto wrapper = static_cast<EmbedderWrapper*>(handle);
    return embedder_embed_ex(handle, text, output, output_size,
                             wrapper->should_normalize ? 1 : 0);
}

CPP_EMBEDDER_API CppEmbedderError embedder_embed_ex(
    EmbedderHandle handle,
    const char* text,
    float* output,
    uint32_t output_size,
    int normalize
) {
    clear_error();

    if (!handle) {
        set_error(CPP_EMBEDDER_ERROR_NULL_POINTER, "Null embedder handle");
        return CPP_EMBEDDER_ERROR_NULL_POINTER;
    }

    if (!text) {
        set_error(CPP_EMBEDDER_ERROR_NULL_POINTER, "Null text pointer");
        return CPP_EMBEDDER_ERROR_NULL_POINTER;
    }

    if (!output) {
        set_error(CPP_EMBEDDER_ERROR_NULL_POINTER, "Null output pointer");
        return CPP_EMBEDDER_ERROR_NULL_POINTER;
    }

    auto wrapper = static_cast<EmbedderWrapper*>(handle);

    if (!wrapper->embedder || !wrapper->embedder->is_loaded()) {
        set_error(CPP_EMBEDDER_ERROR_MODEL_NOT_LOADED, "Model not loaded");
        return CPP_EMBEDDER_ERROR_MODEL_NOT_LOADED;
    }

    uint32_t dim = static_cast<uint32_t>(wrapper->embedder->embedding_dim());
    if (output_size < dim) {
        set_error(CPP_EMBEDDER_ERROR_BUFFER_TOO_SMALL,
                  "Output buffer too small: need " +
                  std::to_string(dim) +
                  ", got " + std::to_string(output_size));
        return CPP_EMBEDDER_ERROR_BUFFER_TOO_SMALL;
    }

    try {
        // Temporarily set normalization config for this call
        bool orig_normalize = wrapper->config.normalize_embeddings;
        wrapper->config.normalize_embeddings = (normalize != 0);

        std::vector<float> embedding = wrapper->embedder->embed(text);

        // Restore original config
        wrapper->config.normalize_embeddings = orig_normalize;

        // Apply normalization if needed (in case config wasn't used)
        if (normalize != 0) {
            cpp_embedder::model::normalize_embedding(embedding);
        }

        std::copy(embedding.begin(), embedding.end(), output);
        return CPP_EMBEDDER_OK;

    } catch (const std::exception& e) {
        set_error(CPP_EMBEDDER_ERROR_COMPUTATION, e.what());
        return CPP_EMBEDDER_ERROR_COMPUTATION;
    }
}

// Batch Text Embedding
CPP_EMBEDDER_API CppEmbedderError embedder_embed_batch(
    EmbedderHandle handle,
    const char** texts,
    uint32_t num_texts,
    float* output,
    uint32_t output_size
) {
    if (!handle) {
        set_error(CPP_EMBEDDER_ERROR_NULL_POINTER, "Null embedder handle");
        return CPP_EMBEDDER_ERROR_NULL_POINTER;
    }

    auto wrapper = static_cast<EmbedderWrapper*>(handle);
    return embedder_embed_batch_ex(handle, texts, num_texts, output, output_size,
                                   wrapper->should_normalize ? 1 : 0);
}

CPP_EMBEDDER_API CppEmbedderError embedder_embed_batch_ex(
    EmbedderHandle handle,
    const char** texts,
    uint32_t num_texts,
    float* output,
    uint32_t output_size,
    int normalize
) {
    clear_error();

    if (!handle) {
        set_error(CPP_EMBEDDER_ERROR_NULL_POINTER, "Null embedder handle");
        return CPP_EMBEDDER_ERROR_NULL_POINTER;
    }

    if (!texts) {
        set_error(CPP_EMBEDDER_ERROR_NULL_POINTER, "Null texts pointer");
        return CPP_EMBEDDER_ERROR_NULL_POINTER;
    }

    if (!output) {
        set_error(CPP_EMBEDDER_ERROR_NULL_POINTER, "Null output pointer");
        return CPP_EMBEDDER_ERROR_NULL_POINTER;
    }

    if (num_texts == 0) {
        return CPP_EMBEDDER_OK; // Nothing to do
    }

    auto wrapper = static_cast<EmbedderWrapper*>(handle);

    if (!wrapper->embedder || !wrapper->embedder->is_loaded()) {
        set_error(CPP_EMBEDDER_ERROR_MODEL_NOT_LOADED, "Model not loaded");
        return CPP_EMBEDDER_ERROR_MODEL_NOT_LOADED;
    }

    uint32_t dim = static_cast<uint32_t>(wrapper->embedder->embedding_dim());
    uint32_t required_size = num_texts * dim;

    if (output_size < required_size) {
        set_error(CPP_EMBEDDER_ERROR_BUFFER_TOO_SMALL,
                  "Output buffer too small: need " +
                  std::to_string(required_size) +
                  ", got " + std::to_string(output_size));
        return CPP_EMBEDDER_ERROR_BUFFER_TOO_SMALL;
    }

    try {
        // Convert C strings to C++ strings
        std::vector<std::string> text_vec;
        text_vec.reserve(num_texts);
        for (uint32_t i = 0; i < num_texts; ++i) {
            if (!texts[i]) {
                set_error(CPP_EMBEDDER_ERROR_NULL_POINTER,
                          "Null text at index " + std::to_string(i));
                return CPP_EMBEDDER_ERROR_NULL_POINTER;
            }
            text_vec.emplace_back(texts[i]);
        }

        std::vector<std::vector<float>> embeddings = wrapper->embedder->embed_batch(text_vec);

        // Copy to output buffer, applying normalization if needed
        for (uint32_t i = 0; i < num_texts; ++i) {
            if (normalize != 0) {
                cpp_embedder::model::normalize_embedding(embeddings[i]);
            }
            std::copy(embeddings[i].begin(), embeddings[i].end(),
                      output + i * dim);
        }

        return CPP_EMBEDDER_OK;

    } catch (const std::exception& e) {
        set_error(CPP_EMBEDDER_ERROR_COMPUTATION, e.what());
        return CPP_EMBEDDER_ERROR_COMPUTATION;
    }
}

// Tokenizer Access
CPP_EMBEDDER_API TokenizerHandle embedder_get_tokenizer(EmbedderHandle handle) {
    if (!handle) return nullptr;
    auto wrapper = static_cast<EmbedderWrapper*>(handle);
    if (!wrapper->embedder) return nullptr;
    return const_cast<cpp_embedder::tokenizer::WordPieceTokenizer*>(
        &wrapper->embedder->tokenizer()
    );
}

CPP_EMBEDDER_API CppEmbedderError tokenizer_encode(
    TokenizerHandle tokenizer,
    const char* text,
    uint32_t* output,
    uint32_t output_size,
    uint32_t* actual_length,
    uint32_t max_length
) {
    clear_error();

    if (!tokenizer) {
        set_error(CPP_EMBEDDER_ERROR_NULL_POINTER, "Null tokenizer handle");
        return CPP_EMBEDDER_ERROR_NULL_POINTER;
    }

    if (!text) {
        set_error(CPP_EMBEDDER_ERROR_NULL_POINTER, "Null text pointer");
        return CPP_EMBEDDER_ERROR_NULL_POINTER;
    }

    if (!output) {
        set_error(CPP_EMBEDDER_ERROR_NULL_POINTER, "Null output pointer");
        return CPP_EMBEDDER_ERROR_NULL_POINTER;
    }

    auto tok = static_cast<cpp_embedder::tokenizer::WordPieceTokenizer*>(tokenizer);
    size_t seq_len = max_length > 0 ? max_length : 256;

    try {
        auto ids = tok->encode(text, seq_len);

        if (output_size < ids.size()) {
            set_error(CPP_EMBEDDER_ERROR_BUFFER_TOO_SMALL,
                      "Output buffer too small: need " +
                      std::to_string(ids.size()) +
                      ", got " + std::to_string(output_size));
            return CPP_EMBEDDER_ERROR_BUFFER_TOO_SMALL;
        }

        for (size_t i = 0; i < ids.size(); ++i) {
            output[i] = static_cast<uint32_t>(ids[i]);
        }

        if (actual_length) {
            *actual_length = static_cast<uint32_t>(ids.size());
        }

        return CPP_EMBEDDER_OK;

    } catch (const std::exception& e) {
        set_error(CPP_EMBEDDER_ERROR_COMPUTATION, e.what());
        return CPP_EMBEDDER_ERROR_COMPUTATION;
    }
}

CPP_EMBEDDER_API CppEmbedderError tokenizer_decode(
    TokenizerHandle tokenizer,
    const uint32_t* token_ids,
    uint32_t num_tokens,
    char* output,
    uint32_t output_size,
    int skip_special
) {
    clear_error();

    if (!tokenizer) {
        set_error(CPP_EMBEDDER_ERROR_NULL_POINTER, "Null tokenizer handle");
        return CPP_EMBEDDER_ERROR_NULL_POINTER;
    }

    if (!token_ids && num_tokens > 0) {
        set_error(CPP_EMBEDDER_ERROR_NULL_POINTER, "Null token_ids pointer");
        return CPP_EMBEDDER_ERROR_NULL_POINTER;
    }

    if (!output) {
        set_error(CPP_EMBEDDER_ERROR_NULL_POINTER, "Null output pointer");
        return CPP_EMBEDDER_ERROR_NULL_POINTER;
    }

    auto tok = static_cast<cpp_embedder::tokenizer::WordPieceTokenizer*>(tokenizer);

    try {
        std::vector<cpp_embedder::tokenizer::Vocab::TokenId> ids;
        ids.reserve(num_tokens);
        for (uint32_t i = 0; i < num_tokens; ++i) {
            ids.push_back(static_cast<cpp_embedder::tokenizer::Vocab::TokenId>(token_ids[i]));
        }

        std::string result = tok->decode(ids);

        if (output_size < result.size() + 1) {
            set_error(CPP_EMBEDDER_ERROR_BUFFER_TOO_SMALL,
                      "Output buffer too small: need " +
                      std::to_string(result.size() + 1) +
                      ", got " + std::to_string(output_size));
            return CPP_EMBEDDER_ERROR_BUFFER_TOO_SMALL;
        }

        std::strcpy(output, result.c_str());
        return CPP_EMBEDDER_OK;

    } catch (const std::exception& e) {
        set_error(CPP_EMBEDDER_ERROR_COMPUTATION, e.what());
        return CPP_EMBEDDER_ERROR_COMPUTATION;
    }
}

CPP_EMBEDDER_API uint32_t tokenizer_cls_token_id(TokenizerHandle tokenizer) {
    if (!tokenizer) return 0;
    auto tok = static_cast<cpp_embedder::tokenizer::WordPieceTokenizer*>(tokenizer);
    return static_cast<uint32_t>(tok->vocab().cls_id());
}

CPP_EMBEDDER_API uint32_t tokenizer_sep_token_id(TokenizerHandle tokenizer) {
    if (!tokenizer) return 0;
    auto tok = static_cast<cpp_embedder::tokenizer::WordPieceTokenizer*>(tokenizer);
    return static_cast<uint32_t>(tok->vocab().sep_id());
}

CPP_EMBEDDER_API uint32_t tokenizer_pad_token_id(TokenizerHandle tokenizer) {
    if (!tokenizer) return 0;
    auto tok = static_cast<cpp_embedder::tokenizer::WordPieceTokenizer*>(tokenizer);
    return static_cast<uint32_t>(tok->vocab().pad_id());
}

CPP_EMBEDDER_API uint32_t tokenizer_unk_token_id(TokenizerHandle tokenizer) {
    if (!tokenizer) return 0;
    auto tok = static_cast<cpp_embedder::tokenizer::WordPieceTokenizer*>(tokenizer);
    return static_cast<uint32_t>(tok->vocab().unk_id());
}

// Utility Functions
CPP_EMBEDDER_API float embedder_cosine_similarity(const float* a, const float* b, uint32_t dim) {
    if (!a || !b || dim == 0) return 0.0f;

    float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;

    for (uint32_t i = 0; i < dim; ++i) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    float denom = std::sqrt(norm_a) * std::sqrt(norm_b);
    if (denom < 1e-8f) return 0.0f;

    return dot / denom;
}

CPP_EMBEDDER_API void embedder_normalize(float* embedding, uint32_t dim) {
    if (!embedding || dim == 0) return;

    float norm_sq = 0.0f;
    for (uint32_t i = 0; i < dim; ++i) {
        norm_sq += embedding[i] * embedding[i];
    }

    if (norm_sq > 1e-12f) {
        float inv_norm = 1.0f / std::sqrt(norm_sq);
        for (uint32_t i = 0; i < dim; ++i) {
            embedding[i] *= inv_norm;
        }
    }
}

CPP_EMBEDDER_API float embedder_norm(const float* embedding, uint32_t dim) {
    if (!embedding || dim == 0) return 0.0f;

    float norm_sq = 0.0f;
    for (uint32_t i = 0; i < dim; ++i) {
        norm_sq += embedding[i] * embedding[i];
    }

    return std::sqrt(norm_sq);
}

CPP_EMBEDDER_API float embedder_dot_product(const float* a, const float* b, uint32_t dim) {
    if (!a || !b || dim == 0) return 0.0f;

    float dot = 0.0f;
    for (uint32_t i = 0; i < dim; ++i) {
        dot += a[i] * b[i];
    }

    return dot;
}

} // extern "C"

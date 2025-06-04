/**
 * @file c_api.h
 * @brief C API for cpp_embedder library
 *
 * This header provides a pure C interface to the cpp_embedder library,
 * enabling bindings from Python (ctypes), Rust (FFI), and other languages.
 *
 * Thread Safety:
 * - All functions are thread-safe after embedder_create() returns
 * - embedder_create() and embedder_destroy() should not be called
 *   concurrently on the same handle
 *
 * Memory Management:
 * - Caller owns output buffers and must ensure sufficient size
 * - EmbedderHandle must be destroyed with embedder_destroy()
 * - Error messages are stored in thread-local storage
 */

#ifndef CPP_EMBEDDER_C_API_H
#define CPP_EMBEDDER_C_API_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Platform-specific export macros */
#ifdef _WIN32
    #ifdef CPP_EMBEDDER_BUILD_SHARED
        #define CPP_EMBEDDER_API __declspec(dllexport)
    #else
        #define CPP_EMBEDDER_API __declspec(dllimport)
    #endif
#else
    #define CPP_EMBEDDER_API __attribute__((visibility("default")))
#endif

/* Opaque handle to the embedder instance */
typedef void* EmbedderHandle;

/* Opaque handle to the tokenizer instance */
typedef void* TokenizerHandle;

/* Error codes */
typedef enum {
    CPP_EMBEDDER_OK = 0,
    CPP_EMBEDDER_ERROR_FILE_NOT_FOUND = 1,
    CPP_EMBEDDER_ERROR_FILE_READ = 2,
    CPP_EMBEDDER_ERROR_FILE_FORMAT = 3,
    CPP_EMBEDDER_ERROR_INVALID_MAGIC = 10,
    CPP_EMBEDDER_ERROR_UNSUPPORTED_VERSION = 11,
    CPP_EMBEDDER_ERROR_WEIGHT_MISMATCH = 12,
    CPP_EMBEDDER_ERROR_CHECKSUM_MISMATCH = 13,
    CPP_EMBEDDER_ERROR_VOCAB_NOT_LOADED = 20,
    CPP_EMBEDDER_ERROR_TOKEN_NOT_FOUND = 21,
    CPP_EMBEDDER_ERROR_SEQUENCE_TOO_LONG = 22,
    CPP_EMBEDDER_ERROR_MODEL_NOT_LOADED = 30,
    CPP_EMBEDDER_ERROR_INVALID_INPUT = 31,
    CPP_EMBEDDER_ERROR_COMPUTATION = 32,
    CPP_EMBEDDER_ERROR_ALLOCATION_FAILED = 40,
    CPP_EMBEDDER_ERROR_INVALID_CONFIG = 50,
    CPP_EMBEDDER_ERROR_NULL_POINTER = 60,
    CPP_EMBEDDER_ERROR_BUFFER_TOO_SMALL = 61,
    CPP_EMBEDDER_ERROR_UNKNOWN = 99
} CppEmbedderError;

/* Model constants */
#define CPP_EMBEDDER_DIM 384
#define CPP_EMBEDDER_VOCAB_SIZE 30522
#define CPP_EMBEDDER_MAX_SEQ_LENGTH 512
#define CPP_EMBEDDER_NUM_LAYERS 6
#define CPP_EMBEDDER_NUM_ATTENTION_HEADS 12

/* =============================================================================
 * Version Information
 * ============================================================================= */

/**
 * Get the library version string.
 * @return Version string (e.g., "1.0.0"). Do not free.
 */
CPP_EMBEDDER_API const char* embedder_version(void);

/**
 * Get the library build info string.
 * @return Build info string. Do not free.
 */
CPP_EMBEDDER_API const char* embedder_build_info(void);

/* =============================================================================
 * Error Handling
 * ============================================================================= */

/**
 * Get the last error message for the current thread.
 * @return Error message string. Do not free. Valid until next API call.
 */
CPP_EMBEDDER_API const char* embedder_get_error(void);

/**
 * Get the last error code for the current thread.
 * @return Error code from CppEmbedderError enum.
 */
CPP_EMBEDDER_API CppEmbedderError embedder_get_error_code(void);

/**
 * Clear the error state for the current thread.
 */
CPP_EMBEDDER_API void embedder_clear_error(void);

/* =============================================================================
 * Embedder Lifecycle
 * ============================================================================= */

/**
 * Create a new embedder instance and load weights.
 *
 * @param weights_path Path to the binary weight file (UTF-8).
 * @return Handle to the embedder, or NULL on failure. Check embedder_get_error().
 */
CPP_EMBEDDER_API EmbedderHandle embedder_create(const char* weights_path);

/**
 * Create a new embedder instance with configuration.
 *
 * @param weights_path Path to the binary weight file (UTF-8). May be NULL.
 * @param max_seq_length Maximum sequence length (1-512). 0 for default (256).
 * @param normalize Whether to L2-normalize embeddings (1=true, 0=false).
 * @param num_threads Number of threads (0=auto-detect).
 * @return Handle to the embedder, or NULL on failure.
 */
CPP_EMBEDDER_API EmbedderHandle embedder_create_with_config(
    const char* weights_path,
    uint32_t max_seq_length,
    int normalize,
    uint32_t num_threads
);

/**
 * Load weights from file into an existing embedder.
 *
 * @param handle Embedder handle.
 * @param weights_path Path to the binary weight file (UTF-8).
 * @return CPP_EMBEDDER_OK on success, error code on failure.
 */
CPP_EMBEDDER_API CppEmbedderError embedder_load(
    EmbedderHandle handle,
    const char* weights_path
);

/**
 * Destroy an embedder instance and free resources.
 *
 * @param handle Embedder handle. Safe to call with NULL.
 */
CPP_EMBEDDER_API void embedder_destroy(EmbedderHandle handle);

/**
 * Check if the embedder model is loaded and ready.
 *
 * @param handle Embedder handle.
 * @return 1 if loaded, 0 if not loaded or handle is NULL.
 */
CPP_EMBEDDER_API int embedder_is_loaded(EmbedderHandle handle);

/* =============================================================================
 * Embedder Properties
 * ============================================================================= */

/**
 * Get the embedding dimension.
 *
 * @param handle Embedder handle.
 * @return Embedding dimension (384), or 0 if handle is NULL.
 */
CPP_EMBEDDER_API uint32_t embedder_get_dim(EmbedderHandle handle);

/**
 * Get the maximum sequence length.
 *
 * @param handle Embedder handle.
 * @return Maximum sequence length, or 0 if handle is NULL.
 */
CPP_EMBEDDER_API uint32_t embedder_get_max_seq_length(EmbedderHandle handle);

/**
 * Get the vocabulary size.
 *
 * @param handle Embedder handle.
 * @return Vocabulary size, or 0 if handle is NULL.
 */
CPP_EMBEDDER_API uint32_t embedder_get_vocab_size(EmbedderHandle handle);

/* =============================================================================
 * Single Text Embedding
 * ============================================================================= */

/**
 * Generate embedding for a single text.
 *
 * @param handle Embedder handle.
 * @param text Input text (UTF-8, null-terminated).
 * @param output Output buffer for embedding. Must have space for embedding_dim floats.
 * @param output_size Size of output buffer in floats.
 * @return CPP_EMBEDDER_OK on success, error code on failure.
 */
CPP_EMBEDDER_API CppEmbedderError embedder_embed(
    EmbedderHandle handle,
    const char* text,
    float* output,
    uint32_t output_size
);

/**
 * Generate embedding for a single text without normalization.
 *
 * @param handle Embedder handle.
 * @param text Input text (UTF-8, null-terminated).
 * @param output Output buffer for embedding.
 * @param output_size Size of output buffer in floats.
 * @param normalize Whether to normalize (1=true, 0=false).
 * @return CPP_EMBEDDER_OK on success, error code on failure.
 */
CPP_EMBEDDER_API CppEmbedderError embedder_embed_ex(
    EmbedderHandle handle,
    const char* text,
    float* output,
    uint32_t output_size,
    int normalize
);

/* =============================================================================
 * Batch Text Embedding
 * ============================================================================= */

/**
 * Generate embeddings for a batch of texts.
 *
 * @param handle Embedder handle.
 * @param texts Array of text pointers (UTF-8, null-terminated).
 * @param num_texts Number of texts in the array.
 * @param output Output buffer. Must have space for num_texts * embedding_dim floats.
 *               Embeddings are stored contiguously in row-major order.
 * @param output_size Size of output buffer in floats.
 * @return CPP_EMBEDDER_OK on success, error code on failure.
 */
CPP_EMBEDDER_API CppEmbedderError embedder_embed_batch(
    EmbedderHandle handle,
    const char** texts,
    uint32_t num_texts,
    float* output,
    uint32_t output_size
);

/**
 * Generate embeddings for a batch of texts with options.
 *
 * @param handle Embedder handle.
 * @param texts Array of text pointers.
 * @param num_texts Number of texts.
 * @param output Output buffer.
 * @param output_size Size of output buffer in floats.
 * @param normalize Whether to normalize embeddings.
 * @return CPP_EMBEDDER_OK on success, error code on failure.
 */
CPP_EMBEDDER_API CppEmbedderError embedder_embed_batch_ex(
    EmbedderHandle handle,
    const char** texts,
    uint32_t num_texts,
    float* output,
    uint32_t output_size,
    int normalize
);

/* =============================================================================
 * Tokenizer Access
 * ============================================================================= */

/**
 * Get the tokenizer handle from an embedder.
 *
 * @param handle Embedder handle.
 * @return Tokenizer handle, or NULL if handle is NULL.
 *         The tokenizer is owned by the embedder; do not destroy separately.
 */
CPP_EMBEDDER_API TokenizerHandle embedder_get_tokenizer(EmbedderHandle handle);

/**
 * Tokenize text and return token IDs.
 *
 * @param tokenizer Tokenizer handle.
 * @param text Input text (UTF-8, null-terminated).
 * @param output Output buffer for token IDs.
 * @param output_size Size of output buffer in uint32_t elements.
 * @param actual_length Output: actual number of tokens (including special tokens).
 * @param max_length Maximum sequence length for padding/truncation.
 * @return CPP_EMBEDDER_OK on success, error code on failure.
 */
CPP_EMBEDDER_API CppEmbedderError tokenizer_encode(
    TokenizerHandle tokenizer,
    const char* text,
    uint32_t* output,
    uint32_t output_size,
    uint32_t* actual_length,
    uint32_t max_length
);

/**
 * Decode token IDs back to text.
 *
 * @param tokenizer Tokenizer handle.
 * @param token_ids Array of token IDs.
 * @param num_tokens Number of token IDs.
 * @param output Output buffer for text.
 * @param output_size Size of output buffer in bytes.
 * @param skip_special Whether to skip special tokens (1=true, 0=false).
 * @return CPP_EMBEDDER_OK on success, error code on failure.
 */
CPP_EMBEDDER_API CppEmbedderError tokenizer_decode(
    TokenizerHandle tokenizer,
    const uint32_t* token_ids,
    uint32_t num_tokens,
    char* output,
    uint32_t output_size,
    int skip_special
);

/**
 * Get special token IDs.
 */
CPP_EMBEDDER_API uint32_t tokenizer_cls_token_id(TokenizerHandle tokenizer);
CPP_EMBEDDER_API uint32_t tokenizer_sep_token_id(TokenizerHandle tokenizer);
CPP_EMBEDDER_API uint32_t tokenizer_pad_token_id(TokenizerHandle tokenizer);
CPP_EMBEDDER_API uint32_t tokenizer_unk_token_id(TokenizerHandle tokenizer);

/* =============================================================================
 * Utility Functions
 * ============================================================================= */

/**
 * Compute cosine similarity between two embeddings.
 *
 * @param a First embedding.
 * @param b Second embedding.
 * @param dim Embedding dimension.
 * @return Cosine similarity in [-1, 1].
 */
CPP_EMBEDDER_API float embedder_cosine_similarity(
    const float* a,
    const float* b,
    uint32_t dim
);

/**
 * L2-normalize an embedding in-place.
 *
 * @param embedding Embedding to normalize.
 * @param dim Embedding dimension.
 */
CPP_EMBEDDER_API void embedder_normalize(float* embedding, uint32_t dim);

/**
 * Compute L2 norm of an embedding.
 *
 * @param embedding Embedding.
 * @param dim Embedding dimension.
 * @return L2 norm.
 */
CPP_EMBEDDER_API float embedder_norm(const float* embedding, uint32_t dim);

/**
 * Compute dot product of two embeddings.
 *
 * @param a First embedding.
 * @param b Second embedding.
 * @param dim Embedding dimension.
 * @return Dot product.
 */
CPP_EMBEDDER_API float embedder_dot_product(
    const float* a,
    const float* b,
    uint32_t dim
);

#ifdef __cplusplus
}
#endif

#endif /* CPP_EMBEDDER_C_API_H */

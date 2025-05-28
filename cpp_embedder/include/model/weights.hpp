#ifndef CPP_EMBEDDER_MODEL_WEIGHTS_HPP
#define CPP_EMBEDDER_MODEL_WEIGHTS_HPP

#include "../math/tensor.hpp"
#include <string>
#include <unordered_map>
#include <vector>
#include <cstdint>
#include <memory>
#include <fstream>

namespace cpp_embedder {
namespace model {

// Weight file format constants
constexpr uint32_t WEIGHT_MAGIC = 0x44424D45; // "EMBD" in little-endian
constexpr uint32_t WEIGHT_VERSION = 1;

// Model weight collection loaded from binary file
class ModelWeights {
public:
    using TensorMap = std::unordered_map<std::string, math::Tensor>;

    // Get tensor by name, throws if not found
    const math::Tensor& get(const std::string& name) const;

    // Check if tensor exists
    bool has(const std::string& name) const;

    // Get all tensor names
    std::vector<std::string> tensor_names() const;

    // Get number of tensors
    size_t size() const { return tensors_.size(); }

    // Get vocabulary (if embedded in weight file)
    const std::vector<std::string>& vocabulary() const { return vocabulary_; }
    bool has_vocabulary() const { return !vocabulary_.empty(); }

private:
    TensorMap tensors_;
    std::vector<std::string> vocabulary_;

    friend class WeightLoader;
};

// Weight loader for binary weight files
// Supports the simplified binary format:
// [4 bytes] magic "EMBD"
// [4 bytes] version (1)
// [4 bytes] num_tensors
// For each tensor:
//   [4 bytes] name length
//   [N bytes] name
//   [4 bytes] num_dims
//   [num_dims * 4 bytes] shape
//   [product(shape) * 4 bytes] float data
// [4 bytes] vocab_size
// For each vocab token:
//   [4 bytes] token length
//   [N bytes] token string
class WeightLoader {
public:
    // Load weights from file path
    static ModelWeights load(const std::string& path);

    // Load weights from input stream
    static ModelWeights load(std::istream& stream);

    // Load weights from memory buffer
    static ModelWeights load_from_memory(const char* data, size_t size);

    // Validate weight file format without fully loading
    static bool validate(const std::string& path);

private:
    static void read_header(std::istream& stream);
    static math::Tensor read_tensor(std::istream& stream);
    static std::string read_string(std::istream& stream);
    static std::vector<std::string> read_vocabulary(std::istream& stream);

    template<typename T>
    static T read_value(std::istream& stream);
};

// Utility to verify all required tensors are present
class WeightValidator {
public:
    static bool validate_bert_weights(const ModelWeights& weights, int num_layers = 6);
    static std::vector<std::string> get_missing_tensors(const ModelWeights& weights, int num_layers = 6);

private:
    static std::vector<std::string> get_required_tensor_names(int num_layers);
};

} // namespace model
} // namespace cpp_embedder

#endif // CPP_EMBEDDER_MODEL_WEIGHTS_HPP

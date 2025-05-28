#include "../../include/model/weights.hpp"
#include <sstream>
#include <cstring>
#include <stdexcept>

namespace cpp_embedder {
namespace model {

using math::Tensor;

// =============================================================================
// ModelWeights Implementation
// =============================================================================

const Tensor& ModelWeights::get(const std::string& name) const {
    auto it = tensors_.find(name);
    if (it == tensors_.end()) {
        throw std::runtime_error("Tensor not found: " + name);
    }
    return it->second;
}

bool ModelWeights::has(const std::string& name) const {
    return tensors_.find(name) != tensors_.end();
}

std::vector<std::string> ModelWeights::tensor_names() const {
    std::vector<std::string> names;
    names.reserve(tensors_.size());
    for (const auto& kv : tensors_) {
        names.push_back(kv.first);
    }
    return names;
}

// =============================================================================
// WeightLoader Implementation
// =============================================================================

template<typename T>
T WeightLoader::read_value(std::istream& stream) {
    T value;
    stream.read(reinterpret_cast<char*>(&value), sizeof(T));
    if (!stream) {
        throw std::runtime_error("Failed to read value from stream");
    }
    return value;
}

std::string WeightLoader::read_string(std::istream& stream) {
    uint32_t length = read_value<uint32_t>(stream);
    if (length > 1024 * 1024) {  // Sanity check: 1MB max
        throw std::runtime_error("String length exceeds maximum");
    }
    std::string str(length, '\0');
    stream.read(&str[0], length);
    if (!stream) {
        throw std::runtime_error("Failed to read string from stream");
    }
    return str;
}

void WeightLoader::read_header(std::istream& stream) {
    // Read and validate magic number
    uint32_t magic = read_value<uint32_t>(stream);
    if (magic != WEIGHT_MAGIC) {
        throw std::runtime_error("Invalid weight file magic number");
    }

    // Read and validate version
    uint32_t version = read_value<uint32_t>(stream);
    if (version != WEIGHT_VERSION) {
        throw std::runtime_error("Unsupported weight file version: " + std::to_string(version));
    }
}

Tensor WeightLoader::read_tensor(std::istream& stream) {
    // Read tensor name
    std::string name = read_string(stream);

    // Read number of dimensions
    uint32_t ndim = read_value<uint32_t>(stream);
    if (ndim == 0 || ndim > 4) {
        throw std::runtime_error("Invalid tensor dimensions: " + std::to_string(ndim));
    }

    // Read shape
    Tensor::Shape shape(ndim);
    size_t total_size = 1;
    for (uint32_t i = 0; i < ndim; ++i) {
        uint32_t dim = read_value<uint32_t>(stream);
        shape[i] = dim;
        total_size *= dim;
    }

    // Read data
    std::vector<float> data(total_size);
    stream.read(reinterpret_cast<char*>(data.data()), total_size * sizeof(float));
    if (!stream) {
        throw std::runtime_error("Failed to read tensor data for: " + name);
    }

    return Tensor(shape, std::move(data));
}

std::vector<std::string> WeightLoader::read_vocabulary(std::istream& stream) {
    uint32_t vocab_size = read_value<uint32_t>(stream);
    if (vocab_size > 100000) {  // Sanity check
        throw std::runtime_error("Vocabulary size exceeds maximum");
    }

    std::vector<std::string> vocabulary;
    vocabulary.reserve(vocab_size);

    for (uint32_t i = 0; i < vocab_size; ++i) {
        vocabulary.push_back(read_string(stream));
    }

    return vocabulary;
}

ModelWeights WeightLoader::load(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open weight file: " + path);
    }
    return load(file);
}

ModelWeights WeightLoader::load(std::istream& stream) {
    ModelWeights weights;

    // Read header
    read_header(stream);

    // Read number of tensors
    uint32_t num_tensors = read_value<uint32_t>(stream);

    // Read tensors
    for (uint32_t i = 0; i < num_tensors; ++i) {
        // Read tensor name first (stored separately from tensor data)
        std::string name = read_string(stream);

        // Read number of dimensions
        uint32_t ndim = read_value<uint32_t>(stream);
        if (ndim == 0 || ndim > 4) {
            throw std::runtime_error("Invalid tensor dimensions: " + std::to_string(ndim));
        }

        // Read shape
        Tensor::Shape shape(ndim);
        size_t total_size = 1;
        for (uint32_t j = 0; j < ndim; ++j) {
            uint32_t dim = read_value<uint32_t>(stream);
            shape[j] = dim;
            total_size *= dim;
        }

        // Read data
        std::vector<float> data(total_size);
        stream.read(reinterpret_cast<char*>(data.data()), total_size * sizeof(float));
        if (!stream) {
            throw std::runtime_error("Failed to read tensor data for: " + name);
        }

        weights.tensors_[name] = Tensor(shape, std::move(data));
    }

    // Read vocabulary
    weights.vocabulary_ = read_vocabulary(stream);

    return weights;
}

ModelWeights WeightLoader::load_from_memory(const char* data, size_t size) {
    std::string buffer(data, size);
    std::istringstream stream(buffer, std::ios::binary);
    return load(stream);
}

bool WeightLoader::validate(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        return false;
    }

    try {
        // Check magic
        uint32_t magic;
        file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
        if (magic != WEIGHT_MAGIC) {
            return false;
        }

        // Check version
        uint32_t version;
        file.read(reinterpret_cast<char*>(&version), sizeof(version));
        if (version != WEIGHT_VERSION) {
            return false;
        }

        return true;
    } catch (...) {
        return false;
    }
}

// =============================================================================
// WeightValidator Implementation
// =============================================================================

std::vector<std::string> WeightValidator::get_required_tensor_names(int num_layers) {
    std::vector<std::string> names;

    // Embedding tensors
    names.push_back("embeddings.word_embeddings.weight");
    names.push_back("embeddings.position_embeddings.weight");
    names.push_back("embeddings.token_type_embeddings.weight");
    names.push_back("embeddings.LayerNorm.weight");
    names.push_back("embeddings.LayerNorm.bias");

    // Encoder layer tensors
    for (int i = 0; i < num_layers; ++i) {
        std::string prefix = "encoder.layer." + std::to_string(i) + ".";

        // Self-attention
        names.push_back(prefix + "attention.self.query.weight");
        names.push_back(prefix + "attention.self.query.bias");
        names.push_back(prefix + "attention.self.key.weight");
        names.push_back(prefix + "attention.self.key.bias");
        names.push_back(prefix + "attention.self.value.weight");
        names.push_back(prefix + "attention.self.value.bias");

        // Attention output
        names.push_back(prefix + "attention.output.dense.weight");
        names.push_back(prefix + "attention.output.dense.bias");
        names.push_back(prefix + "attention.output.LayerNorm.weight");
        names.push_back(prefix + "attention.output.LayerNorm.bias");

        // Feed-forward
        names.push_back(prefix + "intermediate.dense.weight");
        names.push_back(prefix + "intermediate.dense.bias");
        names.push_back(prefix + "output.dense.weight");
        names.push_back(prefix + "output.dense.bias");
        names.push_back(prefix + "output.LayerNorm.weight");
        names.push_back(prefix + "output.LayerNorm.bias");
    }

    return names;
}

bool WeightValidator::validate_bert_weights(const ModelWeights& weights, int num_layers) {
    auto missing = get_missing_tensors(weights, num_layers);
    return missing.empty();
}

std::vector<std::string> WeightValidator::get_missing_tensors(const ModelWeights& weights, int num_layers) {
    std::vector<std::string> missing;
    auto required = get_required_tensor_names(num_layers);

    for (const auto& name : required) {
        if (!weights.has(name)) {
            missing.push_back(name);
        }
    }

    return missing;
}

} // namespace model
} // namespace cpp_embedder

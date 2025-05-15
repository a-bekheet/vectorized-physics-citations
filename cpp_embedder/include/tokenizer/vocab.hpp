#ifndef CPP_EMBEDDER_VOCAB_HPP
#define CPP_EMBEDDER_VOCAB_HPP

#include <string>
#include <unordered_map>
#include <vector>
#include <optional>
#include <cstdint>

namespace cpp_embedder {
namespace tokenizer {

class Vocab {
public:
    using TokenId = int32_t;

    static constexpr const char* CLS_TOKEN = "[CLS]";
    static constexpr const char* SEP_TOKEN = "[SEP]";
    static constexpr const char* PAD_TOKEN = "[PAD]";
    static constexpr const char* UNK_TOKEN = "[UNK]";

    Vocab() = default;

    // Load vocabulary from a text file (one token per line)
    bool load(const std::string& vocab_path);

    // Get the vocabulary size
    size_t size() const;

    // Token to ID lookup (returns UNK ID if not found)
    TokenId token_to_id(const std::string& token) const;

    // ID to token lookup (returns empty optional if invalid)
    std::optional<std::string> id_to_token(TokenId id) const;

    // Check if a token exists in the vocabulary
    bool contains(const std::string& token) const;

    // Special token IDs
    TokenId cls_id() const { return cls_id_; }
    TokenId sep_id() const { return sep_id_; }
    TokenId pad_id() const { return pad_id_; }
    TokenId unk_id() const { return unk_id_; }

    // Check if vocabulary is loaded
    bool is_loaded() const { return !token_to_id_.empty(); }

private:
    std::unordered_map<std::string, TokenId> token_to_id_;
    std::vector<std::string> id_to_token_;

    TokenId cls_id_ = -1;
    TokenId sep_id_ = -1;
    TokenId pad_id_ = -1;
    TokenId unk_id_ = -1;

    void find_special_token_ids();
};

} // namespace tokenizer
} // namespace cpp_embedder

#endif // CPP_EMBEDDER_VOCAB_HPP

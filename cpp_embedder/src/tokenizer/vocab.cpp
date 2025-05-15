#include "tokenizer/vocab.hpp"
#include <fstream>
#include <sstream>

namespace cpp_embedder {
namespace tokenizer {

bool Vocab::load(const std::string& vocab_path) {
    std::ifstream file(vocab_path);
    if (!file.is_open()) {
        return false;
    }

    token_to_id_.clear();
    id_to_token_.clear();

    std::string line;
    TokenId id = 0;

    while (std::getline(file, line)) {
        // Strip trailing whitespace/carriage returns
        while (!line.empty() && (line.back() == '\r' || line.back() == ' ' || line.back() == '\t')) {
            line.pop_back();
        }

        if (line.empty()) {
            continue;
        }

        token_to_id_[line] = id;
        id_to_token_.push_back(line);
        ++id;
    }

    find_special_token_ids();
    return !token_to_id_.empty();
}

size_t Vocab::size() const {
    return id_to_token_.size();
}

Vocab::TokenId Vocab::token_to_id(const std::string& token) const {
    auto it = token_to_id_.find(token);
    if (it != token_to_id_.end()) {
        return it->second;
    }
    return unk_id_;
}

std::optional<std::string> Vocab::id_to_token(TokenId id) const {
    if (id >= 0 && static_cast<size_t>(id) < id_to_token_.size()) {
        return id_to_token_[id];
    }
    return std::nullopt;
}

bool Vocab::contains(const std::string& token) const {
    return token_to_id_.find(token) != token_to_id_.end();
}

void Vocab::find_special_token_ids() {
    auto find_id = [this](const char* token) -> TokenId {
        auto it = token_to_id_.find(token);
        return (it != token_to_id_.end()) ? it->second : -1;
    };

    cls_id_ = find_id(CLS_TOKEN);
    sep_id_ = find_id(SEP_TOKEN);
    pad_id_ = find_id(PAD_TOKEN);
    unk_id_ = find_id(UNK_TOKEN);
}

} // namespace tokenizer
} // namespace cpp_embedder

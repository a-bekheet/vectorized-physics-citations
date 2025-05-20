#include "tokenizer/tokenizer.hpp"
#include <algorithm>
#include <cctype>
#include <sstream>

namespace cpp_embedder {
namespace tokenizer {

WordPieceTokenizer::WordPieceTokenizer(std::shared_ptr<Vocab> vocab)
    : vocab_(std::move(vocab)) {}

std::unique_ptr<WordPieceTokenizer> WordPieceTokenizer::from_vocab_file(const std::string& vocab_path) {
    auto vocab = std::make_shared<Vocab>();
    if (!vocab->load(vocab_path)) {
        return nullptr;
    }
    return std::make_unique<WordPieceTokenizer>(vocab);
}

std::string WordPieceTokenizer::preprocess(const std::string& text) const {
    // Step 1: Lowercase and normalize whitespace
    std::string result;
    result.reserve(text.size());

    bool prev_was_space = true;
    for (char c : text) {
        if (std::isspace(static_cast<unsigned char>(c))) {
            if (!prev_was_space) {
                result += ' ';
                prev_was_space = true;
            }
        } else {
            result += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
            prev_was_space = false;
        }
    }

    // Strip trailing space
    if (!result.empty() && result.back() == ' ') {
        result.pop_back();
    }

    return result;
}

std::vector<std::string> WordPieceTokenizer::split_on_whitespace(const std::string& text) const {
    // Step 2: Split on whitespace to get words
    std::vector<std::string> words;
    std::istringstream stream(text);
    std::string word;

    while (stream >> word) {
        words.push_back(word);
    }

    return words;
}

std::vector<std::string> WordPieceTokenizer::tokenize_word(const std::string& word) const {
    // Step 3: WordPiece algorithm for a single word
    // a. If word is in vocab, use it directly
    // b. Otherwise, greedily match longest prefix in vocab
    // c. Continue with "##" + suffix (WordPiece continuation marker)
    // d. If no match found, use [UNK]

    if (word.empty()) {
        return {};
    }

    // Check if entire word is in vocabulary
    if (vocab_->contains(word)) {
        return {word};
    }

    std::vector<std::string> tokens;
    size_t start = 0;
    bool is_first = true;

    while (start < word.size()) {
        size_t end = word.size();
        std::string matched_token;

        // Greedy longest-match: try progressively shorter substrings
        while (start < end) {
            std::string substr = word.substr(start, end - start);

            // Add continuation prefix for non-first pieces
            if (!is_first) {
                substr = std::string(CONTINUATION_PREFIX) + substr;
            }

            if (vocab_->contains(substr)) {
                matched_token = substr;
                break;
            }

            --end;
        }

        if (matched_token.empty()) {
            // No match found - use [UNK] for entire remaining word
            tokens.push_back(Vocab::UNK_TOKEN);
            break;
        }

        tokens.push_back(matched_token);
        start = end;
        is_first = false;
    }

    return tokens;
}

std::vector<std::string> WordPieceTokenizer::tokenize(const std::string& text) const {
    // Full tokenization pipeline:
    // 1. Preprocess (lowercase, normalize whitespace)
    // 2. Split on whitespace
    // 3. WordPiece tokenize each word
    // Note: Does NOT add [CLS]/[SEP] - that's done in encode()

    if (text.empty()) {
        return {};
    }

    std::string processed = preprocess(text);
    std::vector<std::string> words = split_on_whitespace(processed);

    std::vector<std::string> all_tokens;
    for (const auto& word : words) {
        auto word_tokens = tokenize_word(word);
        all_tokens.insert(all_tokens.end(), word_tokens.begin(), word_tokens.end());
    }

    return all_tokens;
}

std::vector<Vocab::TokenId> WordPieceTokenizer::encode(const std::string& text, size_t max_length) const {
    // Encode with special tokens:
    // Step 4: Add [CLS] at start, [SEP] at end
    // Step 5: Pad or truncate to max_length

    std::vector<Vocab::TokenId> ids;
    ids.reserve(max_length);

    // Add [CLS] token
    ids.push_back(vocab_->cls_id());

    if (!text.empty()) {
        auto tokens = tokenize(text);

        // Reserve space for [CLS] and [SEP]
        size_t max_content_length = max_length >= 2 ? max_length - 2 : 0;

        // Truncate if necessary
        if (tokens.size() > max_content_length) {
            tokens.resize(max_content_length);
        }

        // Convert tokens to IDs
        for (const auto& token : tokens) {
            ids.push_back(vocab_->token_to_id(token));
        }
    }

    // Add [SEP] token
    ids.push_back(vocab_->sep_id());

    // Pad to max_length if needed
    while (ids.size() < max_length) {
        ids.push_back(vocab_->pad_id());
    }

    return ids;
}

std::string WordPieceTokenizer::decode(const std::vector<Vocab::TokenId>& ids) const {
    std::string result;

    for (const auto& id : ids) {
        auto token_opt = vocab_->id_to_token(id);
        if (!token_opt.has_value()) {
            continue;
        }

        const std::string& token = token_opt.value();

        // Skip special tokens in output
        if (token == Vocab::CLS_TOKEN || token == Vocab::SEP_TOKEN || token == Vocab::PAD_TOKEN) {
            continue;
        }

        // Handle continuation tokens (##prefix)
        if (token.size() > 2 && token[0] == '#' && token[1] == '#') {
            result += token.substr(2);
        } else {
            // Add space before regular tokens (except at start)
            if (!result.empty()) {
                result += ' ';
            }
            result += token;
        }
    }

    return result;
}

std::vector<std::vector<Vocab::TokenId>> WordPieceTokenizer::encode_batch(
    const std::vector<std::string>& texts,
    size_t max_length) const {

    std::vector<std::vector<Vocab::TokenId>> batch;
    batch.reserve(texts.size());

    for (const auto& text : texts) {
        batch.push_back(encode(text, max_length));
    }

    return batch;
}

} // namespace tokenizer
} // namespace cpp_embedder

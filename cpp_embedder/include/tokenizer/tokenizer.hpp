#ifndef CPP_EMBEDDER_TOKENIZER_HPP
#define CPP_EMBEDDER_TOKENIZER_HPP

#include "vocab.hpp"
#include <string>
#include <vector>
#include <memory>

namespace cpp_embedder {
namespace tokenizer {

// Tokenizer interface
class ITokenizer {
public:
    virtual ~ITokenizer() = default;

    // Tokenize text into a vector of token strings
    virtual std::vector<std::string> tokenize(const std::string& text) const = 0;

    // Encode text to token IDs with padding/truncation to max_length
    virtual std::vector<Vocab::TokenId> encode(const std::string& text, size_t max_length) const = 0;

    // Decode token IDs back to text
    virtual std::string decode(const std::vector<Vocab::TokenId>& ids) const = 0;

    // Batch encode multiple texts
    virtual std::vector<std::vector<Vocab::TokenId>> encode_batch(
        const std::vector<std::string>& texts,
        size_t max_length) const = 0;
};

// WordPiece tokenizer implementation
class WordPieceTokenizer : public ITokenizer {
public:
    static constexpr size_t DEFAULT_MAX_LENGTH = 512;
    static constexpr const char* CONTINUATION_PREFIX = "##";

    explicit WordPieceTokenizer(std::shared_ptr<Vocab> vocab);

    // Load tokenizer with vocabulary from file
    static std::unique_ptr<WordPieceTokenizer> from_vocab_file(const std::string& vocab_path);

    // ITokenizer interface implementation
    std::vector<std::string> tokenize(const std::string& text) const override;
    std::vector<Vocab::TokenId> encode(const std::string& text, size_t max_length) const override;
    std::string decode(const std::vector<Vocab::TokenId>& ids) const override;
    std::vector<std::vector<Vocab::TokenId>> encode_batch(
        const std::vector<std::string>& texts,
        size_t max_length) const override;

    // Get the underlying vocabulary
    const Vocab& vocab() const { return *vocab_; }

private:
    std::shared_ptr<Vocab> vocab_;

    // Preprocessing: lowercase and normalize whitespace
    std::string preprocess(const std::string& text) const;

    // Split text into words on whitespace
    std::vector<std::string> split_on_whitespace(const std::string& text) const;

    // WordPiece tokenize a single word
    std::vector<std::string> tokenize_word(const std::string& word) const;
};

} // namespace tokenizer
} // namespace cpp_embedder

#endif // CPP_EMBEDDER_TOKENIZER_HPP

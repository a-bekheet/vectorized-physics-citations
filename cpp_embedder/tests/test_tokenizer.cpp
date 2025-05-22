#include "tokenizer/tokenizer.hpp"
#include "tokenizer/vocab.hpp"
#include <iostream>
#include <cassert>
#include <fstream>

using namespace cpp_embedder::tokenizer;

// Helper to create a test vocabulary file
void create_test_vocab(const std::string& path) {
    std::ofstream file(path);
    // Standard BERT special tokens
    file << "[PAD]\n";
    file << "[UNK]\n";
    file << "[CLS]\n";
    file << "[SEP]\n";
    file << "[MASK]\n";
    // Some regular tokens
    file << "hello\n";
    file << "world\n";
    file << "test\n";
    file << "embed\n";
    file << "##ding\n";
    file << "##s\n";
    file << "the\n";
    file << "quick\n";
    file << "brown\n";
    file << "fox\n";
    file << "jump\n";
    file << "##ed\n";
    file << "over\n";
    file << "lazy\n";
    file << "dog\n";
    file.close();
}

void test_vocab() {
    std::cout << "Testing Vocab..." << std::endl;

    create_test_vocab("/tmp/test_vocab.txt");

    Vocab vocab;
    assert(vocab.load("/tmp/test_vocab.txt"));
    assert(vocab.is_loaded());

    // Check size
    assert(vocab.size() == 20);

    // Check special tokens
    assert(vocab.pad_id() == 0);
    assert(vocab.unk_id() == 1);
    assert(vocab.cls_id() == 2);
    assert(vocab.sep_id() == 3);

    // Check token lookup
    assert(vocab.token_to_id("hello") == 5);
    assert(vocab.token_to_id("unknown_token") == vocab.unk_id());

    // Check reverse lookup
    assert(vocab.id_to_token(5).value() == "hello");
    assert(!vocab.id_to_token(100).has_value());

    // Check contains
    assert(vocab.contains("hello"));
    assert(!vocab.contains("nonexistent"));

    std::cout << "Vocab tests passed!" << std::endl;
}

void test_tokenizer() {
    std::cout << "Testing WordPieceTokenizer..." << std::endl;

    create_test_vocab("/tmp/test_vocab.txt");

    auto tokenizer = WordPieceTokenizer::from_vocab_file("/tmp/test_vocab.txt");
    assert(tokenizer != nullptr);

    // Test basic tokenization
    auto tokens = tokenizer->tokenize("Hello World");
    assert(tokens.size() == 2);
    assert(tokens[0] == "hello");
    assert(tokens[1] == "world");

    // Test WordPiece splitting (embeddings -> embed + ##ding + ##s)
    tokens = tokenizer->tokenize("embeddings");
    assert(tokens.size() == 3);
    assert(tokens[0] == "embed");
    assert(tokens[1] == "##ding");
    assert(tokens[2] == "##s");

    // Test unknown token
    tokens = tokenizer->tokenize("xyz");
    assert(tokens.size() == 1);
    assert(tokens[0] == "[UNK]");

    // Test encoding with special tokens
    auto ids = tokenizer->encode("hello world", 10);
    assert(ids.size() == 10);
    assert(ids[0] == tokenizer->vocab().cls_id());  // [CLS]
    assert(ids[1] == 5);  // hello
    assert(ids[2] == 6);  // world
    assert(ids[3] == tokenizer->vocab().sep_id());  // [SEP]
    assert(ids[4] == tokenizer->vocab().pad_id());  // [PAD]

    // Test decoding
    std::string decoded = tokenizer->decode(ids);
    assert(decoded == "hello world");

    // Test empty input
    tokens = tokenizer->tokenize("");
    assert(tokens.empty());

    ids = tokenizer->encode("", 5);
    assert(ids.size() == 5);
    assert(ids[0] == tokenizer->vocab().cls_id());
    assert(ids[1] == tokenizer->vocab().sep_id());

    // Test batch encoding
    std::vector<std::string> texts = {"hello", "world", "test"};
    auto batch = tokenizer->encode_batch(texts, 5);
    assert(batch.size() == 3);
    for (const auto& encoded : batch) {
        assert(encoded.size() == 5);
    }

    std::cout << "WordPieceTokenizer tests passed!" << std::endl;
}

void test_sentence() {
    std::cout << "Testing sentence tokenization..." << std::endl;

    create_test_vocab("/tmp/test_vocab.txt");

    auto tokenizer = WordPieceTokenizer::from_vocab_file("/tmp/test_vocab.txt");

    // Test: "The quick brown fox jumped over the lazy dog"
    auto tokens = tokenizer->tokenize("The quick brown fox jumped over the lazy dog");

    std::cout << "Tokens: ";
    for (const auto& t : tokens) {
        std::cout << "'" << t << "' ";
    }
    std::cout << std::endl;

    // Expected: the, quick, brown, fox, jump, ##ed, over, the, lazy, dog
    assert(tokens.size() == 10);
    assert(tokens[0] == "the");
    assert(tokens[4] == "jump");
    assert(tokens[5] == "##ed");

    std::cout << "Sentence tokenization test passed!" << std::endl;
}

int main() {
    try {
        test_vocab();
        test_tokenizer();
        test_sentence();
        std::cout << "\nAll tests passed!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed: " << e.what() << std::endl;
        return 1;
    }
}

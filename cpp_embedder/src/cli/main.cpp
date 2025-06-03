#include "cli/args.hpp"
#include "model/embedder.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <cstdint>
#include <iomanip>

namespace {

std::vector<std::string> read_texts_from_file(const std::string& path) {
    std::vector<std::string> texts;
    std::ifstream file(path);

    if (!file.is_open()) {
        throw std::runtime_error("Cannot open input file: " + path);
    }

    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty()) {
            texts.push_back(line);
        }
    }

    return texts;
}

std::string escape_json_string(const std::string& s) {
    std::ostringstream result;
    for (char c : s) {
        switch (c) {
            case '"':  result << "\\\""; break;
            case '\\': result << "\\\\"; break;
            case '\b': result << "\\b";  break;
            case '\f': result << "\\f";  break;
            case '\n': result << "\\n";  break;
            case '\r': result << "\\r";  break;
            case '\t': result << "\\t";  break;
            default:
                if ('\x00' <= c && c <= '\x1f') {
                    result << "\\u" << std::hex << std::setfill('0')
                           << std::setw(4) << static_cast<int>(c);
                } else {
                    result << c;
                }
        }
    }
    return result.str();
}

void write_json_output(
    std::ostream& out,
    const std::vector<std::string>& texts,
    const std::vector<std::vector<float>>& embeddings
) {
    out << "{\n  \"embeddings\": [\n";

    for (size_t i = 0; i < texts.size(); ++i) {
        out << "    {\n";
        out << "      \"text\": \"" << escape_json_string(texts[i]) << "\",\n";
        out << "      \"vector\": [";

        const auto& vec = embeddings[i];
        for (size_t j = 0; j < vec.size(); ++j) {
            if (j > 0) out << ", ";
            out << std::setprecision(8) << vec[j];
        }

        out << "]\n    }";
        if (i < texts.size() - 1) {
            out << ",";
        }
        out << "\n";
    }

    out << "  ]\n}\n";
}

void write_binary_output(
    std::ostream& out,
    const std::vector<std::vector<float>>& embeddings
) {
    uint32_t num_embeddings = static_cast<uint32_t>(embeddings.size());
    uint32_t embedding_dim = embeddings.empty() ? 384 : static_cast<uint32_t>(embeddings[0].size());

    out.write(reinterpret_cast<const char*>(&num_embeddings), sizeof(num_embeddings));
    out.write(reinterpret_cast<const char*>(&embedding_dim), sizeof(embedding_dim));

    for (const auto& vec : embeddings) {
        out.write(reinterpret_cast<const char*>(vec.data()), vec.size() * sizeof(float));
    }
}

int run_embedding(
    const cpp_embedder::cli::ParsedArgs& args,
    const std::vector<std::string>& texts
) {
    cpp_embedder::model::EmbedderConfig config;
    config.weights_path = args.model_path;

    cpp_embedder::model::Embedder model(config);

    if (!model.is_loaded()) {
        std::cerr << "Error: Failed to load model from " << args.model_path << "\n";
        return 1;
    }

    std::vector<std::vector<float>> embeddings;

    if (texts.size() == 1) {
        embeddings.push_back(model.embed(texts[0]));
    } else {
        embeddings = model.embed_batch(texts);
    }

    if (args.output_file.has_value()) {
        std::ios_base::openmode mode = std::ios::out;
        if (args.format == cpp_embedder::cli::OutputFormat::BINARY) {
            mode |= std::ios::binary;
        }

        std::ofstream outfile(args.output_file.value(), mode);
        if (!outfile.is_open()) {
            std::cerr << "Error: Cannot open output file: " << args.output_file.value() << "\n";
            return 1;
        }

        if (args.format == cpp_embedder::cli::OutputFormat::JSON) {
            write_json_output(outfile, texts, embeddings);
        } else {
            write_binary_output(outfile, embeddings);
        }
    } else {
        if (args.format == cpp_embedder::cli::OutputFormat::BINARY) {
            write_binary_output(std::cout, embeddings);
        } else {
            write_json_output(std::cout, texts, embeddings);
        }
    }

    return 0;
}

int run_similarity(const cpp_embedder::cli::ParsedArgs& args) {
    cpp_embedder::model::EmbedderConfig config;
    config.weights_path = args.model_path;

    cpp_embedder::model::Embedder model(config);

    if (!model.is_loaded()) {
        std::cerr << "Error: Failed to load model from " << args.model_path << "\n";
        return 1;
    }

    auto embedding1 = model.embed(args.similarity_text1);
    auto embedding2 = model.embed(args.similarity_text2);

    float similarity = cpp_embedder::model::cosine_similarity(embedding1, embedding2);

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Text 1: \"" << args.similarity_text1 << "\"\n";
    std::cout << "Text 2: \"" << args.similarity_text2 << "\"\n";
    std::cout << "Cosine similarity: " << similarity << "\n";

    return 0;
}

} // anonymous namespace

int main(int argc, char* argv[]) {
    if (argc < 2) {
        cpp_embedder::cli::ArgParser::print_usage(argv[0]);
        return 1;
    }

    cpp_embedder::cli::ArgParser parser(argc, argv);
    auto args = parser.parse();

    if (args.has_error) {
        std::cerr << "Error: " << args.error_message << "\n";
        cpp_embedder::cli::ArgParser::print_usage(argv[0]);
        return 1;
    }

    if (args.show_help) {
        cpp_embedder::cli::ArgParser::print_help(argv[0]);
        return 0;
    }

    try {
        if (args.similarity_mode) {
            return run_similarity(args);
        }

        std::vector<std::string> texts = args.texts;

        if (args.input_file.has_value()) {
            auto file_texts = read_texts_from_file(args.input_file.value());
            texts.insert(texts.end(), file_texts.begin(), file_texts.end());
        }

        if (texts.empty()) {
            std::cerr << "Error: No texts to embed\n";
            return 1;
        }

        return run_embedding(args, texts);

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}

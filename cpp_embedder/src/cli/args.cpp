#include "cli/args.hpp"
#include <cstring>
#include <iostream>

namespace cpp_embedder {
namespace cli {

ArgParser::ArgParser(int argc, char* argv[])
    : argc_(argc), argv_(argv), current_index_(1) {}

bool ArgParser::has_more() const {
    return current_index_ < argc_;
}

const char* ArgParser::current() const {
    if (current_index_ < argc_) {
        return argv_[current_index_];
    }
    return nullptr;
}

const char* ArgParser::advance() {
    if (current_index_ < argc_) {
        return argv_[current_index_++];
    }
    return nullptr;
}

const char* ArgParser::peek_next() const {
    if (current_index_ + 1 < argc_) {
        return argv_[current_index_ + 1];
    }
    return nullptr;
}

bool ArgParser::is_option(const char* arg) const {
    return arg != nullptr && arg[0] == '-';
}

bool ArgParser::matches(const char* arg, const char* short_opt, const char* long_opt) const {
    if (arg == nullptr) return false;
    return (short_opt && std::strcmp(arg, short_opt) == 0) ||
           (long_opt && std::strcmp(arg, long_opt) == 0);
}

ParsedArgs ArgParser::error(const std::string& message) {
    ParsedArgs args;
    args.has_error = true;
    args.error_message = message;
    return args;
}

ParsedArgs ArgParser::parse() {
    ParsedArgs args;

    while (has_more()) {
        const char* arg = advance();

        if (matches(arg, "-h", "--help")) {
            args.show_help = true;
            return args;
        }
        else if (matches(arg, "-m", "--model")) {
            const char* value = advance();
            if (value == nullptr || is_option(value)) {
                return error("Option -m/--model requires a path argument");
            }
            args.model_path = value;
        }
        else if (matches(arg, "-t", "--text")) {
            const char* value = advance();
            if (value == nullptr || is_option(value)) {
                return error("Option -t/--text requires a text argument");
            }
            args.texts.push_back(value);
        }
        else if (matches(arg, "-f", "--file")) {
            const char* value = advance();
            if (value == nullptr || is_option(value)) {
                return error("Option -f/--file requires a path argument");
            }
            args.input_file = value;
        }
        else if (matches(arg, "-o", "--output")) {
            const char* value = advance();
            if (value == nullptr || is_option(value)) {
                return error("Option -o/--output requires a path argument");
            }
            args.output_file = value;
        }
        else if (matches(arg, nullptr, "--format")) {
            const char* value = advance();
            if (value == nullptr || is_option(value)) {
                return error("Option --format requires an argument (json or binary)");
            }
            if (std::strcmp(value, "json") == 0) {
                args.format = OutputFormat::JSON;
            } else if (std::strcmp(value, "binary") == 0) {
                args.format = OutputFormat::BINARY;
            } else {
                return error("Invalid format '" + std::string(value) + "'. Use 'json' or 'binary'");
            }
        }
        else if (matches(arg, nullptr, "--similarity")) {
            const char* text1 = advance();
            if (text1 == nullptr || is_option(text1)) {
                return error("Option --similarity requires two text arguments");
            }
            const char* text2 = advance();
            if (text2 == nullptr || is_option(text2)) {
                return error("Option --similarity requires two text arguments");
            }
            args.similarity_mode = true;
            args.similarity_text1 = text1;
            args.similarity_text2 = text2;
        }
        else if (is_option(arg)) {
            return error("Unknown option: " + std::string(arg));
        }
        else {
            return error("Unexpected argument: " + std::string(arg));
        }
    }

    if (!args.show_help && !args.similarity_mode) {
        if (args.model_path.empty()) {
            return error("Model path is required (-m/--model)");
        }
        if (args.texts.empty() && !args.input_file.has_value()) {
            return error("Either -t/--text or -f/--file is required");
        }
    }

    if (args.similarity_mode && args.model_path.empty()) {
        return error("Model path is required for similarity computation (-m/--model)");
    }

    return args;
}

void ArgParser::print_usage(const char* program_name) {
    std::cerr << "Usage: " << program_name << " [options]\n";
    std::cerr << "Try '" << program_name << " --help' for more information.\n";
}

void ArgParser::print_help(const char* program_name) {
    std::cout << "cpp_embed - C++ sentence embedder CLI\n\n";
    std::cout << "Usage: " << program_name << " [options]\n\n";
    std::cout << "Options:\n";
    std::cout << "  -m, --model PATH      Path to weights file (required)\n";
    std::cout << "  -t, --text TEXT       Text to embed (can be repeated)\n";
    std::cout << "  -f, --file PATH       File with texts (one per line)\n";
    std::cout << "  -o, --output PATH     Output file (stdout if not specified)\n";
    std::cout << "  --format FORMAT       Output format: json (default) or binary\n";
    std::cout << "  --similarity T1 T2    Compute cosine similarity between two texts\n";
    std::cout << "  -h, --help            Show this help message\n\n";
    std::cout << "Examples:\n";
    std::cout << "  " << program_name << " -m model.bin -t \"Hello world\"\n";
    std::cout << "  " << program_name << " -m model.bin -f texts.txt -o embeddings.json\n";
    std::cout << "  " << program_name << " -m model.bin -t \"first\" -t \"second\" --format json\n";
    std::cout << "  " << program_name << " -m model.bin --similarity \"hello\" \"hi there\"\n\n";
    std::cout << "Output Formats:\n";
    std::cout << "  json    - JSON with text and vector fields\n";
    std::cout << "  binary  - Binary format with header and float32 vectors\n";
}

} // namespace cli
} // namespace cpp_embedder

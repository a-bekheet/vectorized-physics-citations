#ifndef CPP_EMBEDDER_CLI_ARGS_HPP
#define CPP_EMBEDDER_CLI_ARGS_HPP

#include <string>
#include <vector>
#include <optional>

namespace cpp_embedder {
namespace cli {

enum class OutputFormat {
    JSON,
    BINARY
};

struct ParsedArgs {
    std::string model_path;
    std::vector<std::string> texts;
    std::optional<std::string> input_file;
    std::optional<std::string> output_file;
    OutputFormat format = OutputFormat::JSON;
    bool similarity_mode = false;
    std::string similarity_text1;
    std::string similarity_text2;
    bool show_help = false;
    std::string error_message;
    bool has_error = false;
};

class ArgParser {
public:
    ArgParser(int argc, char* argv[]);

    ParsedArgs parse();

    static void print_usage(const char* program_name);
    static void print_help(const char* program_name);

private:
    int argc_;
    char** argv_;
    int current_index_;

    bool has_more() const;
    const char* current() const;
    const char* advance();
    const char* peek_next() const;

    bool is_option(const char* arg) const;
    bool matches(const char* arg, const char* short_opt, const char* long_opt) const;

    ParsedArgs error(const std::string& message);
};

} // namespace cli
} // namespace cpp_embedder

#endif // CPP_EMBEDDER_CLI_ARGS_HPP

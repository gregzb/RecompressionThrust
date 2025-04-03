#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "recompression.hpp"
#include "rlslp.hpp"
#include "types.hpp"
#include "util.hpp"

#include "mio/mio.hpp"
#include <argparse/argparse.hpp>

std::vector<symbol_t> bcomp(std::vector<symbol_t> &input, std::vector<Rlslp::Rule> &rules, std::vector<level_t> &levels,
                            level_t level) {
    std::vector<symbol_t> output;

    symbol_t next_rule = rules.size();
    symbol_t current_symbol = input[0];
    symbol_t current_len = 1;

    std::unordered_map<std::pair<symbol_t, symbol_t>, symbol_t, pair_hash<symbol_t, symbol_t>> block_to_rule;

    auto process_last_symbol = [&]() {
        if (current_len == 1) {
            output.push_back(current_symbol);
            return;
        }
        auto block = std::make_pair(current_symbol, current_len);
        auto it = block_to_rule.find(block);
        if (it == block_to_rule.end()) {
            rules.push_back({.block = {.symbol = current_symbol, .count = current_len}});
            levels.push_back(level);
            auto [new_it, _insertion_happened] = block_to_rule.insert({block, next_rule++});
            it = new_it;
        }
        output.push_back(it->second);
    };

    for (symbol_t i = 1; i < input.size(); i++) {
        if (input[i - 1] == input[i]) {
            current_len++;
            continue;
        }
        process_last_symbol();
        current_symbol = input[i];
        current_len = 1;
    }
    process_last_symbol();
    return output;
}

// In theory this shouldn't be fixed but it's easier to reason about deterministic programs
std::mt19937 rng(7);
std::uniform_int_distribution<std::mt19937::result_type> dist16(0, (1 << 16) - 1);

std::vector<symbol_t> pcomp(std::vector<symbol_t> &input, std::vector<Rlslp::Rule> &rules, std::vector<level_t> &levels,
                            level_t level) {
    std::vector<symbol_t> output;

    symbol_t next_rule = rules.size();

    std::vector<bool> rand_bits(rules.size());
    for (symbol_t i = 0; i < rand_bits.size(); i += 16) {
        symbol_t rand_int = dist16(rng);
        for (symbol_t j = 0; j < 16 && i + j < rand_bits.size(); j++) {
            rand_bits[i + j] = rand_int & 1;
            rand_int >>= 1;
        }
    }

    std::unordered_map<std::pair<symbol_t, symbol_t>, symbol_t, pair_hash<symbol_t, symbol_t>> pair_to_rule;

    for (symbol_t i = 0; i < input.size();) {
        if (i < input.size() - 1 && !rand_bits[input[i]] && rand_bits[input[i + 1]]) {
            auto pair = std::make_pair(input[i], input[i + 1]);
            auto it = pair_to_rule.find(pair);
            if (it == pair_to_rule.end()) {
                rules.push_back({.pair = {
                                     .children = {input[i], input[i + 1]},
                                 }});
                levels.push_back(level);
                auto [new_it, _insertion_happened] = pair_to_rule.insert({pair, next_rule++});
                it = new_it;
            }
            output.push_back(it->second);
            i += 2;
        } else {
            output.push_back(input[i]);
            i += 1;
        }
    }

    return output;
}

Rlslp recompression(symbol_t alphabet_size, std::vector<symbol_t> &input) {
    Rlslp rlslp{.rules = std::vector<Rlslp::Rule>(256, Rlslp::Rule{}),
                .levels = std::vector<level_t>(256, 0),
                .alphabet_size = 256,
                .root = -1,
                .len = (symbol_t)input.size()};

    level_t level = 1;
    while (input.size() > 1) {
        input = bcomp(input, rlslp.rules, rlslp.levels, level);
        level++;

        if (input.size() == 1)
            continue;

        std::vector<symbol_t> next_input;
        do {
            next_input = pcomp(input, rlslp.rules, rlslp.levels, level);
        } while (input.size() == next_input.size());
        input = next_input;
        level++;
    }

    rlslp.root = rlslp.rules.size() - 1;
    return rlslp;
}

std::vector<symbol_t> expand(const Rlslp &rlslp) {
    std::vector<symbol_t> out;
    auto dfs = [&](auto self, int32_t root) -> void {
        if (root < rlslp.alphabet_size) {
            out.push_back(root);
            return;
        }
        auto &rule = rlslp.rules[root];
        if (Rlslp::level_is_pair(rlslp.levels[root])) {
            self(self, rule.pair.left());
            self(self, rule.pair.right());
        } else {
            for (int32_t i = 0; i < rule.block.count; i++) {
                self(self, rule.block.symbol);
            }
        }
    };
    dfs(dfs, rlslp.root);
    return out;
}

struct TraversalItem {
    symbol_t node;
    symbol_t next_child_index;
};

template <typename F> void expand(const Rlslp &rlslp, F write_f) {
    std::vector<TraversalItem> stack;

    stack.push_back({.node = rlslp.root, .next_child_index = 0});

    std::array<char, 256> buf{};
    size_t filled = 0;

    auto add_to_buf = [&](symbol_t node) {
        buf[filled++] = node;
        if (filled == buf.size()) {
            write_f(buf.data(), filled);
            filled = 0;
        }
    };

    while (!stack.empty()) {
        TraversalItem top = stack.back();
        stack.pop_back();

        symbol_t node = top.node;
        if (node < rlslp.alphabet_size) {
            add_to_buf(node);
            continue;
        }
        const auto &rule = rlslp.rules[node];
        if (Rlslp::level_is_pair(rlslp.levels[node])) {
            if (top.next_child_index < 1) {
                stack.push_back({.node = node, .next_child_index = top.next_child_index + 1});
            }
            stack.push_back({.node = rule.pair.children[top.next_child_index], .next_child_index = 0});
        } else {
            if (top.next_child_index < rule.block.count - 1) {
                stack.push_back({.node = node, .next_child_index = top.next_child_index + 1});
            }
            stack.push_back({.node = rule.block.symbol, .next_child_index = 0});
        }
    }

    if (filled > 0) {
        write_f(buf.data(), filled);
    }
}

std::string generate_dot(const Rlslp &rlslp) {
    std::stringstream out;
    out << "digraph {\n";

    std::vector<bool> used_terminals(rlslp.alphabet_size, false);
    std::vector<int> indeg(rlslp.rules.size(), 0);

    auto check_and_mark_terminal = [&](symbol_t symbol) {
        if (symbol < rlslp.alphabet_size) {
            used_terminals[symbol] = true;
        }
    };

    for (symbol_t i = rlslp.alphabet_size; i < rlslp.rules.size(); i++) {
        auto &rule = rlslp.rules[i];
        if (Rlslp::level_is_pair(rlslp.levels[i])) {
            out << "    " << i << " -> " << rule.pair.left() << "[tailport=sw]" << "\n";
            check_and_mark_terminal(rule.pair.left());
            indeg[rule.pair.left()]++;
            out << "    " << i << " -> " << rule.pair.right() << "[tailport=se]" << "\n";
            check_and_mark_terminal(rule.pair.right());
            indeg[rule.pair.right()]++;
        } else {
            for (int32_t j = 0; j < rule.block.count; j++) {
                out << "    " << i << " -> " << rule.block.symbol << "\n";
                check_and_mark_terminal(rule.block.symbol);
                indeg[rule.block.symbol]++;
            }
        }
    }

    symbol_t rank_start = rlslp.alphabet_size;

    int cnt = 0;

    auto group_upto = [&](symbol_t upto) {
        out << "{\n";
        out << "rank=same\n";
        for (symbol_t i = rank_start; i < upto; i++) {
            out << "    " << i;
            if (indeg[i] > 1) {
                out << "[style = filled;fillcolor = lightgreen;]";
                cnt++;
            }
            out << "\n";
        }
        out << "}\n";
    };

    // Assumes that rules are written in level order
    // Doesn't have to be this way, I'm just lazy
    for (symbol_t i = rlslp.alphabet_size + 1; i < rlslp.rules.size(); i++) {
        auto &prev_rule = rlslp.rules[i - 1];
        auto &rule = rlslp.rules[i];
        if (rlslp.levels[i - 1] != rlslp.levels[i]) {
            group_upto(i);
            rank_start = i;
        }
    }

    group_upto(rlslp.rules.size());

    out << "\n";

    out << "{\n";
    out << "rank=same\n";
    for (symbol_t i = 0; i < rlslp.alphabet_size; i++) {
        if (!used_terminals[i])
            continue;
        out << "    " << i << "[label=\"" << escapeChar(i) << "\"]\n";
    }
    out << "}\n";

    out << "}\n";

    std::cout << "generating dot: nodes with indeg > 1: " << cnt << std::endl;
    std::cout << "generating dot: total nodes: " << (indeg.size() - rlslp.alphabet_size) << std::endl;

    return out.str();
};

auto read_file(const std::string &file_path) {
    std::error_code error;
    mio::ummap_source read_only_file = mio::make_mmap<mio::ummap_source>(file_path, 0, mio::map_entire_file, error);
    if (error) {
        std::cout << "Failed to read file at " << file_path << ", exiting!" << std::endl;
        std::cout << error.message() << std::endl;
        exit(error.value());
    }
    return read_only_file;
}

int main(int argc, char **argv) {
    argparse::ArgumentParser program("recompression");

    argparse::ArgumentParser compress_command("compress");
    compress_command.add_description("The primary mode for this program. Takes a normal file and applies recompression "
                                     "to it, outputting the compressed format to a new file");
    std::string mode;
    std::string input_filename;
    std::string output_filename;
    std::string debug_dot_file;
    compress_command.add_argument("--mode")
        .help("Select how to run recompression. Options: cpu, thrust-cpu, thrust-gpu. Defaults to thrust-gpu")
        .default_value("thrust-gpu")
        .store_into(mode);
    compress_command.add_argument("--generate-debug-dot")
        .help("Specify the file to put the generate dot graph into. Defaults to no file")
        .default_value("")
        .store_into(debug_dot_file);
    compress_command.add_argument("input").help("What file to compress").store_into(input_filename);
    compress_command.add_argument("output")
        .help("Name of the file to output the compressed format into")
        .store_into(output_filename);

    argparse::ArgumentParser decompress_command("decompress");
    decompress_command.add_description(
        "This format isn't really intended to be decompressed, but it feels incomplete without it");
    decompress_command.add_argument("input").help("What file to decompress").store_into(input_filename);
    decompress_command.add_argument("output")
        .help("Name of the file to output the decompressed file into")
        .store_into(output_filename);
    program.add_subparser(compress_command);
    program.add_subparser(decompress_command);
    try {
        program.parse_args(argc, argv);
    } catch (const std::exception &err) {
        std::cerr << "Failed to parse args" << std::endl;
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        std::exit(1);
    }

    if (program.is_subcommand_used("compress")) {
        auto read_only_file = read_file(input_filename);
        auto process_rlslp = [&](const Rlslp &rlslp) {
            if (debug_dot_file != "") {
                time_f_print_void(
                    [&]() {
                        auto dot_str = generate_dot(rlslp);
                        {
                            std::ofstream dot_out(debug_dot_file);
                            dot_out << dot_str;
                        }
                    },
                    "generate dot file");
            }

            time_f_print_void([&]() { rlslp.serialize_to(output_filename); }, "write rlslp to disk");
        };
        if (mode == "cpu") {
            auto rlslp = time_f_print(
                [&]() {
                    auto vec_to_compress = time_f_print(
                        [&]() { return std::vector<symbol_t>(read_only_file.begin(), read_only_file.end()); },
                        "read file and transfer to a convenient spot in memory");
                    return recompression(256, vec_to_compress);
                },
                "plain cpu recompression w/ file read");
            process_rlslp(rlslp);
        } else if (mode == "thrust-cpu") {
            auto rlslp = time_f_print(
                [&]() { return Cu::Thrust<false>::recompression(256, read_only_file.begin(), read_only_file.end()); },
                "thrust-cpu recompression w/ file read");
            process_rlslp(rlslp);
        } else if (mode == "thrust-gpu") {
            time_f_print_void([&]() { Cu::init(); }, "cuda init");
            auto rlslp = time_f_print(
                [&]() { return Cu::Thrust<true>::recompression(256, read_only_file.begin(), read_only_file.end()); },
                "thrust-gpu recompression w/ file read");
            process_rlslp(rlslp);
        } else {
            std::cout << "Unknown mode: " << mode << std::endl;
            exit(1);
        }
    } else {
        std::ofstream file_out(output_filename, std::ios::binary);
        auto rlslp = time_f_print([&]() { return Rlslp::of_serialized(input_filename); }, "deserialize rlslp");
        time_f_print_void([&]() { expand(rlslp, [&](const char *buf, size_t size) { file_out.write(buf, size); }); },
                          "decompress rlslp and write to file");
    }
    return 0;
}
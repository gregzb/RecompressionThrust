#include <iostream>
#include <vector>
#include <array>
#include <unordered_map>
#include <utility>
#include <algorithm>
#include <string>
#include <random>
#include <sstream>
#include <cctype>
#include <fstream>
#include <chrono>

#include "types.hpp"
#include "util.hpp"
#include "recompression.hpp"

#include <argparse/argparse.hpp>

struct Rlslp
{
    struct Pair
    {
        std::array<symbol_t, 2> children;
        symbol_t &left()
        {
            return children[0];
        }
        const symbol_t &left() const
        {
            return children[0];
        }
        symbol_t &right()
        {
            return children[1];
        }
        const symbol_t &right() const
        {
            return children[1];
        }
    };

    struct Block
    {
        symbol_t symbol;
        symbol_t count;
    };

    struct Rule
    {
        union
        {
            Pair pair;
            Block block;
        };
        level_t level;
    };

    std::vector<Rule> rules;
    symbol_t alphabet_size;
    symbol_t root;
    len_t len;

    static bool level_is_block(level_t level)
    {
        return level % 2 == 1;
    }

    static bool level_is_pair(level_t level)
    {
        return level % 2 == 0;
    }
};

std::vector<symbol_t> bcomp(std::vector<symbol_t> &input, std::vector<Rlslp::Rule> &rules, level_t level)
{
    // std::cout << "bcomp level: " << level << std::endl;
    std::vector<symbol_t> output;

    symbol_t next_rule = rules.size();
    symbol_t current_symbol = input[0];
    symbol_t current_len = 1;

    std::unordered_map<std::pair<symbol_t, symbol_t>, symbol_t, pair_hash<symbol_t, symbol_t>> block_to_rule;

    auto process_last_symbol = [&]()
    {
        if (current_len == 1)
        {
            output.push_back(current_symbol);
            return;
        }
        auto block = std::make_pair(current_symbol, current_len);
        auto it = block_to_rule.find(block);
        if (it == block_to_rule.end())
        {
            rules.push_back({.block = {
                                 .symbol = current_symbol,
                                 .count = current_len},
                             .level = level});
            auto [new_it, _insertion_happened] = block_to_rule.insert({block, next_rule++});
            it = new_it;
        }
        output.push_back(it->second);
    };

    for (len_t i = 1; i < input.size(); i++)
    {
        if (input[i - 1] == input[i])
        {
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

// std::random_device dev;
// std::mt19937 rng(dev());
std::mt19937 rng(7);
std::uniform_int_distribution<std::mt19937::result_type> dist16(0, (1 << 16) - 1);

std::vector<symbol_t> pcomp(std::vector<symbol_t> &input, std::vector<Rlslp::Rule> &rules, level_t level)
{
    // std::cout << "pcomp level: " << level << std::endl;
    std::vector<symbol_t> output;

    symbol_t next_rule = rules.size();

    std::vector<bool> rand_bits(rules.size());
    for (symbol_t i = 0; i < rand_bits.size(); i += 16)
    {
        symbol_t rand_int = dist16(rng);
        for (symbol_t j = 0; j < 16 && i + j < rand_bits.size(); j++)
        {
            rand_bits[i + j] = rand_int & 1;
            rand_int >>= 1;
        }
    }

    std::unordered_map<std::pair<symbol_t, symbol_t>, symbol_t, pair_hash<symbol_t, symbol_t>> pair_to_rule;

    for (symbol_t i = 0; i < input.size();)
    {
        if (i < input.size() - 1 && !rand_bits[input[i]] && rand_bits[input[i + 1]])
        {
            auto pair = std::make_pair(input[i], input[i + 1]);
            auto it = pair_to_rule.find(pair);
            if (it == pair_to_rule.end())
            {
                rules.push_back({.pair = {
                                     .children = {input[i], input[i + 1]},
                                 },
                                 .level = level});
                auto [new_it, _insertion_happened] = pair_to_rule.insert({pair, next_rule++});
                it = new_it;
            }
            output.push_back(it->second);
            i += 2;
        }
        else
        {
            output.push_back(input[i]);
            i += 1;
        }
    }

    return output;
}

Rlslp recompression(symbol_t alphabet_size, std::vector<symbol_t> input)
{
    Rlslp rlslp{
        .rules = std::vector<Rlslp::Rule>(256, Rlslp::Rule{}),
        .alphabet_size = 256,
        .root = -1,
        .len = (len_t)input.size()};

    level_t level = 1;
    while (input.size() > 1)
    {
        input = bcomp(input, rlslp.rules, level);
        level++;

        if (input.size() == 1)
            continue;

        std::vector<symbol_t> next_input;
        do
        {
            next_input = pcomp(input, rlslp.rules, level);
        } while (input.size() == next_input.size());
        input = next_input;
        level++;

        std::cout << input.size() << std::endl;
    }

    std::cout << "layers: " << level << std::endl;

    rlslp.root = rlslp.rules.size() - 1;
    return rlslp;
}

std::vector<symbol_t> expand(const Rlslp &rlslp)
{
    std::vector<symbol_t> out;
    auto dfs = [&](auto self, int32_t root) -> void
    {
        if (root < rlslp.alphabet_size)
        {
            out.push_back(root);
            return;
        }
        auto &rule = rlslp.rules[root];
        if (Rlslp::level_is_pair(rule.level))
        {
            self(self, rule.pair.left());
            self(self, rule.pair.right());
        }
        else
        {
            for (int32_t i = 0; i < rule.block.count; i++)
            {
                self(self, rule.block.symbol);
            }
        }
    };
    dfs(dfs, rlslp.root);
    return out;
}

std::string generate_dot(const Rlslp &rlslp)
{
    std::stringstream out;
    out << "digraph {\n";

    std::vector<bool> used_terminals(rlslp.alphabet_size, false);
    std::vector<int> indeg(rlslp.rules.size(), 0);

    auto check_and_mark_terminal = [&](symbol_t symbol)
    {
        if (symbol < rlslp.alphabet_size)
        {
            used_terminals[symbol] = true;
        }
    };

    for (symbol_t i = rlslp.alphabet_size; i < rlslp.rules.size(); i++)
    {
        auto &rule = rlslp.rules[i];
        if (Rlslp::level_is_pair(rule.level))
        {
            out << "    " << i << " -> " << rule.pair.left() << "[tailport=sw]" << "\n";
            check_and_mark_terminal(rule.pair.left());
            indeg[rule.pair.left()]++;
            out << "    " << i << " -> " << rule.pair.right() << "[tailport=se]" << "\n";
            check_and_mark_terminal(rule.pair.right());
            indeg[rule.pair.right()]++;
        }
        else
        {
            for (int32_t j = 0; j < rule.block.count; j++)
            {
                out << "    " << i << " -> " << rule.block.symbol << "\n";
                check_and_mark_terminal(rule.block.symbol);
                indeg[rule.block.symbol]++;
            }
        }
    }

    symbol_t rank_start = rlslp.alphabet_size;

    int cnt = 0;

    auto group_upto = [&](symbol_t upto)
    {
        out << "{\n";
        out << "rank=same\n";
        for (symbol_t i = rank_start; i < upto; i++)
        {
            out << "    " << i;
            if (indeg[i] > 1)
            {
                out << "[style = filled;fillcolor = lightgreen;]";
                cnt++;
            }
            out << "\n";
        }
        out << "}\n";
    };

    for (symbol_t i = rlslp.alphabet_size + 1; i < rlslp.rules.size(); i++)
    {
        auto &prev_rule = rlslp.rules[i - 1];
        auto &rule = rlslp.rules[i];
        if (prev_rule.level != rule.level)
        {
            group_upto(i);
            rank_start = i;
        }
    }

    group_upto(rlslp.rules.size());

    out << "\n";

    out << "{\n";
    out << "rank=same\n";
    for (symbol_t i = 0; i < rlslp.alphabet_size; i++)
    {
        if (!used_terminals[i])
            continue;
        out << "    " << i << "[label=\"" << escapeChar(i) << "\"]\n";
    }
    out << "}\n";

    out << "}\n";

    std::cout << "indeg > 1: " << cnt << std::endl;

    return out.str();
};

std::string read_file_into_string(const std::string &filePath)
{
    std::ifstream file(filePath, std::ios::in | std::ios::binary | std::ios::ate);
    if (!file.is_open())
    {
        throw std::runtime_error("Failed to open file: " + filePath);
    }

    // Determine file size.
    std::streamsize fileSize = file.tellg();
    // Move to the start of the file.
    file.seekg(0, std::ios::beg);

    // Create a string to hold the data.
    std::string buffer(fileSize, '\0');

    // Read all characters into the string.
    if (!file.read(&buffer[0], fileSize))
    {
        throw std::runtime_error("Failed to read file: " + filePath);
    }

    return buffer; // Return the file contents as a string.
}

int main(int argc, char **argv)
{
    argparse::ArgumentParser program("recompression");
    std::string mode;
    std::string input_filename;
    std::string output_filename;
    program.add_argument("--mode").help("Select how to run recompression. Options: cpu, thrust-cpu, thrust-gpu. Defaults to thrust-gpu").default_value("thrust-gpu").store_into(mode);
    program.add_argument("input").help("What file to compress").store_into(input_filename);
    program.add_argument("output").help("Name of the file to output the compressed format into").store_into(output_filename);
    try
    {
        program.parse_args(argc, argv);
    }
    catch (const std::exception &err)
    {
        std::cerr << "Failed to parse args" << std::endl;
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        std::exit(1);
    }

    auto text_to_compress = read_file_into_string(input_filename);
    auto vec_to_compress = to_vec(text_to_compress);
    using std::chrono::duration;
    using std::chrono::duration_cast;
    using std::chrono::high_resolution_clock;
    using std::chrono::nanoseconds;
    Cu::init();
    // auto t1 = high_resolution_clock::now();
    // auto rlslp = recompression(256, vec_to_compress);
    if (mode == "cpu")
    {
        recompression(256, vec_to_compress);
    }
    else if (mode == "thrust-cpu")
    {
        Cu::Thrust<false>::recompression(256, vec_to_compress);
    }
    else if (mode == "thrust-gpu")
    {
        Cu::Thrust<true>::recompression(256, vec_to_compress);
    }
    else
    {
        std::cout << "Unknown mode: " << mode << std::endl;
        exit(1);
    }
    // auto t2 = high_resolution_clock::now();
    // duration<double, std::milli> ms_double = t2 - t1;
    // std::cout << "total " << (ms_double.count()) << " ms\n";
    // auto expanded = expand(rlslp);
    // auto expanded_text = to_string(expanded);
    // // std::cout << text_to_compress << std::endl;
    // // std::cout << expanded_text << std::endl;

    // auto dot_str = generate_dot(rlslp);
    // {
    //     std::ofstream dot_out("debug.dot");
    //     dot_out << dot_str;
    // }
    // std::cout
    //     << "Used " << (rlslp.rules.size() - rlslp.alphabet_size) << " nonterminals" << std::endl;
    return 0;
}
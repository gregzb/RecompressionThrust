#pragma once
#include <array>
#include <vector>
#include <fstream>
#include "types.hpp"

struct Rlslp
{
    struct Pair
    {
        std::array<symbol_t, 2> children;
        inline symbol_t &left() { return children[0]; }
        inline const symbol_t &left() const { return children[0]; }
        inline symbol_t &right() { return children[1]; }
        inline const symbol_t &right() const { return children[1]; }
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
    };

    std::vector<Rule> rules;
    std::vector<level_t> levels;
    symbol_t alphabet_size;
    symbol_t root;
    len_t len;

    inline static bool level_is_block(level_t level) { return level % 2 == 1; }

    inline static bool level_is_pair(level_t level) { return level % 2 == 0; }

    void serialize_to(const std::string &file_name) const
    {
        std::ofstream file_out(file_name, std::ios::binary);
        symbol_t rules_size = rules.size();
        file_out.write(reinterpret_cast<const char *>(&rules_size), sizeof(symbol_t));
        file_out.write(reinterpret_cast<const char *>(rules.data()), sizeof(Rule) * rules.size());
        file_out.write(reinterpret_cast<const char *>(levels.data()), sizeof(level_t) * rules.size());
        file_out.write(reinterpret_cast<const char *>(&alphabet_size), sizeof(symbol_t));
        file_out.write(reinterpret_cast<const char *>(&root), sizeof(symbol_t));
        file_out.write(reinterpret_cast<const char *>(&len), sizeof(len_t));
    }

    static Rlslp of_serialized(const std::string &file_name)
    {
        std::ifstream file_in(file_name, std::ios::binary);
        symbol_t rules_size;
        file_in.read(reinterpret_cast<char *>(&rules_size), sizeof(symbol_t));
        auto rlslp = Rlslp{
            .rules = std::vector<Rule>(rules_size),
            .levels = std::vector<level_t>(rules_size),
            .alphabet_size = -1,
            .root = -1,
            .len = -1};

        file_in.read(reinterpret_cast<char *>(rlslp.rules.data()), sizeof(Rule) * rlslp.rules.size());
        file_in.read(reinterpret_cast<char *>(rlslp.levels.data()), sizeof(level_t) * rlslp.rules.size());
        file_in.read(reinterpret_cast<char *>(&rlslp.alphabet_size), sizeof(symbol_t));
        file_in.read(reinterpret_cast<char *>(&rlslp.root), sizeof(symbol_t));
        file_in.read(reinterpret_cast<char *>(&rlslp.len), sizeof(len_t));
        return rlslp;
    }
};